"""Unified command-line interface for the Chicago Crime Analytics Platform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chicago_crimes.analytics.benchmark import run_distributed_benchmark
from chicago_crimes.analytics.profiling import (
    profile_dataset,
    run_block_stats,
    run_correlation_analysis,
)
from chicago_crimes.analytics.registry import ALL_TASKS
from chicago_crimes.analytics.runner import run_analytical_tasks
from chicago_crimes.core.config import AppConfig, load_config
from chicago_crimes.core.logging import get_logger, setup_logging
from chicago_crimes.core.spark import spark_session
from chicago_crimes.experiments import (
    run_famd_analysis,
    run_imbalance_analysis,
    run_pca_analysis,
    run_sampling_analysis,
)
from chicago_crimes.ingestion.bronze import ingest_csv_to_bronze
from chicago_crimes.ingestion.csv_loader import load_crime_csv, load_raw_csv_for_bronze
from chicago_crimes.processing.cleaning import run_legacy_preprocessing
from chicago_crimes.processing.gold import build_gold_tables
from chicago_crimes.processing.silver import run_silver_pipeline
from chicago_crimes.storage.delta import delta_history, merge_silver_delta
from chicago_crimes.storage.lakehouse import read_table, sync_to_delta
from chicago_crimes.storage.local import is_remote_uri, write_parquet
from chicago_crimes.storage.minio import ensure_bucket, s3a_uri

logger = get_logger("cli")


def _resolve_input_format(path: str | Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    target = Path(path)
    return "csv" if target.is_file() and target.suffix.lower() == ".csv" else "parquet"


def _load_input_dataframe(
    spark: Any,
    path: str | Path,
    fmt: str,
    sep: str,
    limit: int | None = None,
    *,
    raw_strings: bool = False,
):
    resolved = _resolve_input_format(path, fmt)
    if resolved == "csv":
        if raw_strings:
            return load_raw_csv_for_bronze(spark, path, sep=sep, limit=limit)
        return load_crime_csv(spark, path, sep=sep, limit=limit)
    df = read_table(spark, path, table_format="delta" if resolved == "delta" else "auto")
    return df.limit(limit) if limit is not None else df


def _default_analytics_input(cfg: AppConfig) -> str:
    if Path(cfg.paths.silver_dir).exists():
        return cfg.paths.silver_dir
    return cfg.paths.legacy_clean_dir


def _handle_ingest(args: argparse.Namespace, cfg: AppConfig) -> int:
    if args.overwrite and args.append:
        raise ValueError("Cannot pass both --overwrite and --append.")
    mode = "overwrite" if args.overwrite else ("append" if args.append else "errorifexists")
    out_path = Path(args.output or cfg.paths.bronze_dir)
    if mode == "errorifexists" and out_path.exists():
        raise FileExistsError(
            f"Output path already exists ({out_path}). Pass --overwrite or --append explicitly."
        )

    csv_input = args.input or cfg.paths.raw_csv
    with spark_session(cfg, app_name="ChicagoCrimes-Ingest") as spark:
        if args.layer == "raw":
            df = load_crime_csv(spark, csv_input, sep=args.sep, limit=args.limit)
            written = write_parquet(df, out_path, mode=mode)
            logger.info("Raw typed dataset written to %s", written)
        else:
            result = ingest_csv_to_bronze(
                spark,
                csv_path=csv_input,
                bronze_dir=out_path,
                sep=args.sep,
                limit=args.limit,
                source_id=args.source_id,
                batch_id=args.batch_id,
                mode=mode,
            )
            logger.info(
                "Bronze ingest complete -> %s (rows=%s corrupt=%s batch=%s)",
                result.output_path,
                result.row_count,
                result.corrupt_count,
                result.batch_id,
            )
    return 0


def _handle_process(args: argparse.Namespace, cfg: AppConfig) -> int:
    mode = "overwrite" if args.overwrite else "errorifexists"

    if args.pipeline == "legacy":
        out_path = Path(args.output or cfg.paths.legacy_clean_dir)
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output path already exists ({out_path}). Pass --overwrite to replace explicitly."
            )
        report_path = (
            Path(args.report_path)
            if args.report_path
            else Path(cfg.paths.stats_output_dir) / "structural_anomalies_report.txt"
        )
        with spark_session(cfg, app_name="ChicagoCrimes-Process-Legacy") as spark:
            raw_df = _load_input_dataframe(
                spark,
                args.input or cfg.paths.raw_csv,
                fmt=args.input_format,
                sep=args.sep,
                limit=args.limit,
            )
            processed_df = run_legacy_preprocessing(raw_df, report_path=report_path)
            written = write_parquet(processed_df, out_path, mode=mode)
            logger.info("Legacy processed dataset written to %s", written)
        return 0

    out_path = Path(args.output or cfg.paths.silver_dir)
    if args.quarantine_dir:
        quarantine_path = Path(args.quarantine_dir)
    elif args.output:
        quarantine_path = out_path.parent / f"{out_path.name}_quarantine"
    else:
        quarantine_path = Path(cfg.paths.quarantine_dir)

    default_input = (
        cfg.paths.bronze_dir if Path(cfg.paths.bronze_dir).exists() else cfg.paths.raw_csv
    )
    input_path = args.input or default_input

    with spark_session(cfg, app_name="ChicagoCrimes-Process-Silver") as spark:
        source_df = _load_input_dataframe(
            spark,
            input_path,
            fmt=args.input_format,
            sep=args.sep,
            limit=args.limit,
            raw_strings=True,
        )
        result = run_silver_pipeline(
            source_df,
            silver_dir=out_path,
            quarantine_dir=quarantine_path,
            superseded_dir=args.superseded_dir,
            manifest_path=args.manifest_path,
            mode=mode,
            deduplicate=not args.no_deduplicate,
            quarantine_out_of_bounds_geo=args.quarantine_oob_geo,
        )
        logger.info(
            "Silver dataset written to %s (accepted=%s superseded=%s quarantined=%s)",
            result.silver_path,
            result.accounting.accepted_records,
            result.accounting.superseded_records,
            result.accounting.quarantined_records,
        )
    return 0


def _handle_gold(args: argparse.Namespace, cfg: AppConfig) -> int:
    input_path = args.input or cfg.paths.silver_dir
    if is_remote_uri(input_path):
        cfg.spark.enable_delta = True
    out_dir = Path(args.output or cfg.paths.gold_dir)
    mode = "overwrite" if args.overwrite else "errorifexists"
    with spark_session(cfg, app_name="ChicagoCrimes-Gold") as spark:
        silver_df = read_table(spark, input_path, table_format="auto")
        result = build_gold_tables(silver_df, out_dir, mode=mode)
        logger.info(
            "Gold aggregates written to %s (%d tables from %d Silver rows)",
            result.gold_dir,
            len(result.tables),
            result.source_silver_count,
        )
    return 0


def _handle_analyze(args: argparse.Namespace, cfg: AppConfig) -> int:
    if args.list:
        for task in ALL_TASKS:
            print(
                f"{task.task_id}\t{task.category:<10}\t{task.domain_name:<42}\t{task.legacy_name}"
            )
        return 0

    input_path = args.input or _default_analytics_input(cfg)
    if args.format == "delta" or args.version is not None or is_remote_uri(input_path):
        cfg.spark.enable_delta = True
    out_dir = Path(args.output or cfg.paths.analytics_output_dir)
    with spark_session(cfg, app_name="ChicagoCrimes-Analyze") as spark:
        df = read_table(spark, input_path, table_format=args.format, version_as_of=args.version)
        results = run_analytical_tasks(
            df,
            output_dir=out_dir,
            tasks=args.tasks,
            category=args.category,
            config=cfg.analytics,
            render_plots=not args.no_plots,
        )
        logger.info("Completed %d analytical task(s) -> %s", len(results), out_dir)
    return 0


def _handle_profile(args: argparse.Namespace, cfg: AppConfig) -> int:
    input_path = args.input or _default_analytics_input(cfg)
    if args.input_format == "delta" or is_remote_uri(input_path):
        cfg.spark.enable_delta = True
    out_dir = Path(args.output or cfg.paths.stats_output_dir)
    with spark_session(cfg, app_name="ChicagoCrimes-Profile") as spark:
        df = _load_input_dataframe(spark, input_path, fmt=args.input_format, sep=args.sep)
        kinds = {"summary", "correlation", "block"} if args.kind == "all" else {args.kind}
        if "summary" in kinds:
            profile_dataset(
                df,
                out_dir=out_dir,
                dataset_name=args.dataset_name,
                source_path=input_path,
                cache_df=True,
            )
        if "correlation" in kinds:
            run_correlation_analysis(df, out_dir=out_dir, threshold=args.corr_threshold)
        if "block" in kinds:
            run_block_stats(df, out_dir=out_dir / "block_stats")
    logger.info("Profiling artifacts written to %s", out_dir)
    return 0


def _handle_lakehouse(args: argparse.Namespace, cfg: AppConfig) -> int:
    if args.action == "ensure-bucket":
        bucket = ensure_bucket(cfg.storage, bucket_name=args.bucket)
        print(json.dumps({"bucket": bucket, "endpoint": cfg.storage.s3_endpoint}, indent=2))
        return 0

    cfg.spark.enable_delta = True
    if cfg.storage.backend == "s3a":
        ensure_bucket(cfg.storage)

    target = args.target or s3a_uri(cfg.storage.s3_bucket, "silver/chicago_crimes")

    if args.action == "sync-delta":
        src = args.input or cfg.paths.silver_dir
        mode = "overwrite" if args.overwrite else "errorifexists"
        with spark_session(cfg, app_name="ChicagoCrimes-Lakehouse-Sync") as spark:
            res = sync_to_delta(
                spark,
                source_path=src,
                target_delta_path=target,
                limit=args.limit,
                mode=mode,
                partition_by=args.partition_by,
            )
        print(json.dumps(res, indent=2))
        return 0

    if args.action == "merge-silver":
        src = args.input or cfg.paths.silver_dir
        with spark_session(cfg, app_name="ChicagoCrimes-Lakehouse-Merge") as spark:
            src_df = read_table(spark, src, table_format="auto")
            if args.limit is not None:
                src_df = src_df.limit(args.limit)
            res = merge_silver_delta(spark, src_df, target_path=target)
        print(json.dumps(res, indent=2))
        return 0

    if args.action == "history":
        with spark_session(cfg, app_name="ChicagoCrimes-Lakehouse-History") as spark:
            hist = delta_history(spark, target, limit=args.limit or 10)
        print(json.dumps(hist, indent=2))
        return 0

    raise ValueError(f"Unsupported lakehouse action: {args.action}")


def _handle_benchmark(args: argparse.Namespace, cfg: AppConfig) -> int:
    if not args.no_delta:
        cfg.spark.enable_delta = True
    input_path = args.input or _default_analytics_input(cfg)
    out_json = Path(args.output or (Path(cfg.paths.analytics_output_dir) / "benchmark_report.json"))
    report = run_distributed_benchmark(
        cfg,
        input_path=input_path,
        output_json=out_json,
        partition_counts=args.partitions,
        sample_limit=args.limit,
        write_delta_output=not args.no_delta,
    )
    print(json.dumps(report, indent=2))
    return 0


def _handle_experiment(args: argparse.Namespace, cfg: AppConfig) -> int:
    input_path = args.input or _default_analytics_input(cfg)
    if is_remote_uri(input_path):
        cfg.spark.enable_delta = True
    out_root = Path(args.output or cfg.paths.experiments_output_dir)
    exp_cfg = cfg.experiments
    kinds = {"pca", "famd", "imbalance", "sampling"} if args.kind == "all" else {args.kind}

    with spark_session(cfg, app_name=f"ChicagoCrimes-Experiment-{args.kind}") as spark:
        df = read_table(spark, input_path, table_format="auto")
        if "pca" in kinds:
            run_pca_analysis(
                df,
                out_dir=out_root / "pca",
                n_components=exp_cfg.pca_components,
                sample_fraction=args.sample_fraction or exp_cfg.pca_sample_fraction,
                sample_max_rows=args.sample_max_rows or exp_cfg.pca_sample_max_rows,
                seed=exp_cfg.seed,
            )
        if "famd" in kinds:
            run_famd_analysis(
                df,
                out_dir=out_root / "famd",
                sample_fraction=args.sample_fraction or exp_cfg.famd_sample_fraction,
                sample_max_rows=args.sample_max_rows or exp_cfg.famd_sample_max_rows,
                n_components=exp_cfg.famd_components,
                seed=exp_cfg.seed,
            )
        if "imbalance" in kinds:
            run_imbalance_analysis(df, out_dir=out_root / "imbalance")
        if "sampling" in kinds:
            run_sampling_analysis(
                df,
                out_dir=out_root / "sampling",
                seed=exp_cfg.seed,
                pca_sample_max_rows=args.sample_max_rows or exp_cfg.sampling_pca_max_rows,
                write_sampled_dataset=not args.no_write_sampled,
            )
    logger.info("Experiment outputs saved to %s", out_root)
    return 0


def _handle_ingest_soda(args: argparse.Namespace, cfg: AppConfig) -> int:
    from chicago_crimes.ingestion.kafka_producer import ingest_soda_to_kafka

    summary = ingest_soda_to_kafka(
        cfg.streaming,
        watermark_path=args.watermark_file,
        since=args.since,
        until=args.until,
        max_records=args.limit,
        page_size=args.page_size,
        max_pages=args.max_pages,
        default_lookback_hours=args.lookback_hours,
        publish_kafka=args.publish_kafka,
        commit_watermark=not args.no_commit_watermark,
        output_jsonl=args.output_jsonl,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _handle_stream(args: argparse.Namespace, cfg: AppConfig) -> int:
    from chicago_crimes.streaming.crime_stream import run_crime_stream

    summary = run_crime_stream(
        cfg,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        checkpoint_location=args.checkpoint,
        bronze_target=args.bronze_target,
        silver_target=args.silver_target,
        quarantine_target=args.quarantine_target,
        dlq_target=args.dlq_target,
        metrics_log_path=args.metrics_log,
        starting_offsets=args.starting_offsets,
        available_now=args.available_now,
        trigger_seconds=args.trigger_seconds,
        max_batches=args.max_batches,
        timeout_seconds=args.timeout_seconds,
        partition_by=args.partition_by,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _handle_validate(args: argparse.Namespace, cfg: AppConfig) -> int:
    from chicago_crimes.processing.ge_validation import run_ge_validation_suite

    silver_input = args.silver_input or cfg.paths.silver_dir
    gold_input = args.gold_input or cfg.paths.gold_dir
    if args.table_format == "delta" or is_remote_uri(silver_input) or is_remote_uri(gold_input):
        cfg.spark.enable_delta = True
    output_path = args.output or "artifacts/quality/ge_validation_report.json"

    with spark_session(cfg, app_name="ChicagoCrimes-Validate") as spark:
        report = run_ge_validation_suite(
            spark,
            layer=args.layer,
            silver_input=silver_input,
            gold_input=gold_input,
            table_format=args.table_format,
            sample_limit=args.sample_limit,
            output_path=output_path,
            raise_on_failure=False,
        )
    print(json.dumps({"success": report["success"], "output_path": output_path}, indent=2))
    return 0 if report["success"] else 1


def _handle_ml(args: argparse.Namespace, cfg: AppConfig) -> int:
    from chicago_crimes.ml.models import run_forecasting_experiment

    gold_input = args.gold_input or cfg.paths.gold_dir
    if (
        args.table_format == "delta"
        or is_remote_uri(gold_input)
        or (args.silver_input is not None and is_remote_uri(args.silver_input))
    ):
        cfg.spark.enable_delta = True
    output_dir = args.output or cfg.paths.ml_output_dir
    with spark_session(cfg, app_name="ChicagoCrimes-ML") as spark:
        bundle = run_forecasting_experiment(
            spark,
            gold_dir=gold_input,
            silver_input=args.silver_input,
            table_format=args.table_format,
            output_dir=output_dir,
            experiment_name=args.experiment_name,
            min_district_days=args.min_district_days,
            max_history_days=args.max_history_days,
            include_spark_models=not args.no_spark_models,
        )
    print(
        json.dumps(
            {
                "champion_model": bundle.champion_model,
                "champion_test_metrics": bundle.report["champion_test_metrics"],
                "beats_seasonal_naive": bundle.report["champion_beats_seasonal_naive"],
                "mlflow_tracking_uri": bundle.tracking_uri,
                "report_path": str(bundle.report_path),
            },
            indent=2,
        )
    )
    return 0


def _handle_publish_bigquery(args: argparse.Namespace, cfg: AppConfig) -> int:
    from chicago_crimes.storage.bigquery import publish_gold_and_silver_to_bigquery

    dry_run = not args.execute
    gold_input = args.gold_input or cfg.paths.gold_dir
    if (
        args.table_format == "delta"
        or is_remote_uri(gold_input)
        or (args.silver_input is not None and is_remote_uri(args.silver_input))
    ):
        cfg.spark.enable_delta = True
    tables = tuple(args.tables) if args.tables else (
        "daily_citywide_counts",
        "daily_district_counts",
        "category_distribution",
        "monthly_crime_summary",
        "district_community_summary",
    )
    with spark_session(cfg, app_name="ChicagoCrimes-PublishBQ") as spark:
        manifest = publish_gold_and_silver_to_bigquery(
            spark,
            project_id=args.project,
            dataset_id=args.dataset,
            gold_dir=gold_input,
            silver_input=args.silver_input,
            table_format=args.table_format,
            tables=tables,
            location=args.location,
            dry_run=dry_run,
            output_manifest=args.output or "artifacts/bigquery/publish_manifest.json",
        )
    print(
        json.dumps(
            {
                "project_id": manifest["project_id"],
                "dataset_id": manifest["dataset_id"],
                "dry_run": manifest["dry_run"],
                "table_count": manifest["table_count"],
                "output_manifest": args.output or "artifacts/bigquery/publish_manifest.json",
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chicago-crimes",
        description="Chicago Crime Analytics Distributed Engineering & ML Platform",
    )
    parser.add_argument(
        "--profile",
        choices=["local", "standalone"],
        default=None,
        help="Execution profile overriding YAML configuration.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML override file path.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_p = subparsers.add_parser("ingest", help="Ingest raw Chicago Crimes CSV into Bronze.")
    ingest_p.add_argument("--input", default=None, help="Path to source CSV file.")
    ingest_p.add_argument("--output", default=None, help="Destination Parquet directory.")
    ingest_p.add_argument(
        "--layer",
        choices=["bronze", "raw"],
        default="bronze",
        help="Ingest as non-destructive string-preserving Bronze or legacy typed raw Parquet.",
    )
    ingest_p.add_argument("--source-id", default=None, help="Source identifier for Bronze lineage.")
    ingest_p.add_argument("--batch-id", default=None, help="Explicit ingestion batch identifier.")
    ingest_p.add_argument("--sep", default=";", help="CSV delimiter (default ';').")
    ingest_p.add_argument(
        "--limit", type=int, default=None, help="Optional row limit for smoke runs."
    )
    ingest_p.add_argument(
        "--append", action="store_true", help="Append to existing destination Parquet dataset."
    )
    ingest_p.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting existing destination path."
    )

    proc_p = subparsers.add_parser("process", help="Run Bronze->Silver or legacy preprocessing.")
    proc_p.add_argument(
        "--pipeline",
        choices=["silver", "legacy"],
        default="silver",
        help="Select validated Silver pipeline (default) or legacy preprocessing.",
    )
    proc_p.add_argument("--input", default=None, help="Input Bronze Parquet or CSV path.")
    proc_p.add_argument(
        "--input-format",
        choices=["auto", "parquet", "csv"],
        default="auto",
        help="Format of --input dataset (auto detects .csv vs Parquet).",
    )
    proc_p.add_argument("--output", default=None, help="Destination Silver/processed Parquet dir.")
    proc_p.add_argument("--quarantine-dir", default=None, help="Quarantine Parquet directory.")
    proc_p.add_argument("--superseded-dir", default=None, help="Superseded duplicate Parquet dir.")
    proc_p.add_argument("--manifest-path", default=None, help="Silver accounting JSON path.")
    proc_p.add_argument("--sep", default=";", help="CSV delimiter (default ';').")
    proc_p.add_argument(
        "--limit", type=int, default=None, help="Optional row limit for smoke runs."
    )
    proc_p.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deterministic (ID, Updated On) deduplication in Silver.",
    )
    proc_p.add_argument(
        "--quarantine-oob-geo",
        action="store_true",
        help="Quarantine out-of-bounds coordinates instead of preserving incident with null geo.",
    )
    proc_p.add_argument("--report-path", default=None, help="Legacy anomaly summary text path.")
    proc_p.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting existing destination path."
    )

    gold_p = subparsers.add_parser(
        "gold", help="Build reproducible Gold analytical aggregates from Silver."
    )
    gold_p.add_argument("--input", default=None, help="Validated Silver Parquet directory path.")
    gold_p.add_argument("--output", default=None, help="Destination Gold root directory.")
    gold_p.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting existing Gold tables."
    )

    ana_p = subparsers.add_parser(
        "analyze", help="Execute registered analytical tasks (AQ-01..AQ-24)."
    )
    ana_p.add_argument("--input", default=None, help="Processed Parquet directory path.")
    ana_p.add_argument("--output", default=None, help="Output directory for CSV/PNG/manifest.")
    ana_p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific task IDs, legacy names, or domain names to run.",
    )
    ana_p.add_argument(
        "--category",
        choices=["temporal", "geographic", "categories", "arrests"],
        default=None,
        help="Run all tasks in a category.",
    )
    ana_p.add_argument(
        "--format",
        choices=["auto", "parquet", "delta"],
        default="auto",
        help="Input table format (auto detects Delta vs Parquet).",
    )
    ana_p.add_argument(
        "--version",
        type=int,
        default=None,
        help="Optional Delta table version number for time-travel queries.",
    )
    ana_p.add_argument(
        "--no-plots", action="store_true", help="Skip rendering matplotlib PNG charts."
    )
    ana_p.add_argument(
        "--list", action="store_true", help="List all 24 registered analytical tasks and exit."
    )

    prof_p = subparsers.add_parser(
        "profile", help="Generate dataset statistics, correlation matrix, or block stats."
    )
    prof_p.add_argument("--input", default=None, help="Input dataset path.")
    prof_p.add_argument(
        "--input-format",
        choices=["auto", "parquet", "delta", "csv"],
        default="parquet",
        help="Format of --input dataset.",
    )
    prof_p.add_argument("--sep", default=";", help="CSV delimiter when --input-format=csv.")
    prof_p.add_argument("--output", default=None, help="Directory for profiling outputs.")
    prof_p.add_argument(
        "--kind",
        choices=["summary", "correlation", "block", "all"],
        default="summary",
        help="Profiling workload to run.",
    )
    prof_p.add_argument(
        "--dataset-name", default="dataset", help="Label for stats report filename."
    )
    prof_p.add_argument(
        "--corr-threshold",
        type=float,
        default=0.9,
        help="Absolute Pearson correlation threshold for high_correlations.txt.",
    )

    exp_p = subparsers.add_parser(
        "experiment", help="Run PCA, FAMD, imbalance, or sampling experiments."
    )
    exp_p.add_argument("--input", default=None, help="Processed Parquet directory path.")
    exp_p.add_argument(
        "--output", default=None, help="Root output directory for experiment artifacts."
    )
    exp_p.add_argument(
        "--kind",
        required=True,
        choices=["pca", "famd", "imbalance", "sampling", "all"],
        help="Experiment suite to execute.",
    )
    exp_p.add_argument(
        "--sample-fraction", type=float, default=None, help="Override sample fraction."
    )
    exp_p.add_argument(
        "--sample-max-rows", type=int, default=None, help="Override max sampled rows."
    )
    exp_p.add_argument(
        "--no-write-sampled",
        action="store_true",
        help="Skip persisting sampled Parquet datasets in sampling experiments.",
    )

    lake_p = subparsers.add_parser(
        "lakehouse", help="Manage MinIO buckets, Delta tables, ACID MERGEs, and history."
    )
    lake_p.add_argument(
        "action",
        choices=["ensure-bucket", "sync-delta", "merge-silver", "history"],
        help="Lakehouse operation to execute.",
    )
    lake_p.add_argument("--bucket", default=None, help="Target MinIO bucket override.")
    lake_p.add_argument("--input", default=None, help="Source Parquet/Delta dataset path.")
    lake_p.add_argument("--target", default=None, help="Target Delta path or s3a:// URI.")
    lake_p.add_argument(
        "--partition-by",
        nargs="+",
        default=None,
        help="Optional Delta partition columns (e.g., year month).",
    )
    lake_p.add_argument("--limit", type=int, default=None, help="Optional row or history limit.")
    lake_p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing Delta target in sync-delta."
    )

    bench_p = subparsers.add_parser(
        "benchmark", help="Run distributed Spark & Delta execution benchmarks."
    )
    bench_p.add_argument("--input", default=None, help="Source Silver Parquet/Delta path.")
    bench_p.add_argument("--output", default=None, help="Output JSON report path.")
    bench_p.add_argument(
        "--partitions",
        nargs="+",
        type=int,
        default=[8, 16, 32],
        help="Shuffle partition counts to benchmark.",
    )
    bench_p.add_argument("--limit", type=int, default=None, help="Optional row sample limit.")
    bench_p.add_argument(
        "--no-delta",
        action="store_true",
        help="Skip writing Delta summary tables during benchmark sweeps.",
    )

    soda_p = subparsers.add_parser(
        "ingest-soda",
        help="Fetch incremental Chicago Open Data (SODA API v2) records with watermarks.",
    )
    soda_p.add_argument("--since", default=None, help="Override start ISO timestamp.")
    soda_p.add_argument("--until", default=None, help="Optional upper bound ISO timestamp.")
    soda_p.add_argument("--limit", type=int, default=None, help="Maximum records to fetch.")
    soda_p.add_argument(
        "--page-size",
        type=int,
        default=None,
        help="SODA page size per HTTP request (default from config: 1000).",
    )
    soda_p.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum SODA pages to fetch in one bounded catch-up run (default: 10).",
    )
    soda_p.add_argument(
        "--lookback-hours",
        type=int,
        default=72,
        help="Default lookback window in hours when no watermark or --since is set.",
    )
    soda_p.add_argument(
        "--watermark-file",
        default=None,
        help="Path to SODA JSON watermark cursor file.",
    )
    soda_p.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional local JSONL path for captured incremental Bronze records.",
    )
    soda_p.add_argument(
        "--publish-kafka",
        action="store_true",
        help="Publish fetched Bronze events idempotently to Kafka before saving watermark.",
    )
    soda_p.add_argument("--bootstrap-servers", default=None, help="Kafka bootstrap servers.")
    soda_p.add_argument("--topic", default=None, help="Target Kafka topic.")
    soda_p.add_argument(
        "--no-commit-watermark",
        action="store_true",
        help="Fetch without updating the persisted watermark JSON file.",
    )

    stream_p = subparsers.add_parser(
        "stream",
        help="Consume Kafka crime events via Spark Structured Streaming into Bronze/Silver Delta.",
    )
    stream_p.add_argument("--bootstrap-servers", default=None, help="Kafka bootstrap servers.")
    stream_p.add_argument("--topic", default=None, help="Kafka source topic.")
    stream_p.add_argument("--checkpoint", default=None, help="Streaming checkpoint path/URI.")
    stream_p.add_argument("--bronze-target", default=None, help="Streaming Bronze Delta target.")
    stream_p.add_argument("--silver-target", default=None, help="Streaming Silver Delta target.")
    stream_p.add_argument(
        "--quarantine-target", default=None, help="Silver quarantine Delta target."
    )
    stream_p.add_argument("--dlq-target", default=None, help="Malformed JSON DLQ Delta target.")
    stream_p.add_argument("--metrics-log", default=None, help="Batch accounting JSONL output path.")
    stream_p.add_argument(
        "--starting-offsets", default=None, help="Kafka startingOffsets (earliest/latest)."
    )
    stream_p.add_argument(
        "--available-now",
        action="store_true",
        help="Drain currently available Kafka offsets with trigger(availableNow=True) and exit.",
    )
    stream_p.add_argument("--trigger-seconds", type=int, default=None, help="Micro-batch interval.")
    stream_p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional non-empty micro-batch limit before graceful stop.",
    )
    stream_p.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional query timeout in seconds.",
    )
    stream_p.add_argument(
        "--partition-by",
        nargs="+",
        default=["year"],
        help="Silver Delta partition columns.",
    )

    val_p = subparsers.add_parser(
        "validate", help="Run Great Expectations quality suites on Silver and Gold tables."
    )
    val_p.add_argument(
        "--layer",
        choices=["silver", "gold", "both"],
        default="both",
        help="Target data layers to validate.",
    )
    val_p.add_argument("--silver-input", default=None, help="Silver Parquet/Delta path or URI.")
    val_p.add_argument("--gold-input", default=None, help="Gold tables root directory.")
    val_p.add_argument(
        "--table-format",
        choices=["auto", "parquet", "delta"],
        default="auto",
        help="Silver storage format.",
    )
    val_p.add_argument(
        "--sample-limit",
        type=int,
        default=100_000,
        help="Max Silver rows to evaluate with Great Expectations.",
    )
    val_p.add_argument(
        "--output",
        default=None,
        help="Path for JSON quality report.",
    )

    ml_p = subparsers.add_parser(
        "ml", help="Train chronological crime forecasting models and track runs in MLflow."
    )
    ml_p.add_argument("--gold-input", default=None, help="Gold tables directory.")
    ml_p.add_argument(
        "--silver-input", default=None, help="Optional Silver fallback if Gold is absent."
    )
    ml_p.add_argument(
        "--table-format",
        choices=["auto", "parquet", "delta"],
        default="auto",
        help="Input storage format for Silver fallback.",
    )
    ml_p.add_argument("--output", default=None, help="MLflow & chart output directory.")
    ml_p.add_argument(
        "--experiment-name",
        default="chicago_crimes_daily_forecast",
        help="MLflow experiment name.",
    )
    ml_p.add_argument(
        "--min-district-days",
        type=int,
        default=30,
        help="Minimum distinct dates required per district.",
    )
    ml_p.add_argument(
        "--max-history-days",
        type=int,
        default=None,
        help="Optional rolling trailing day horizon to limit training history.",
    )
    ml_p.add_argument(
        "--no-spark-models",
        action="store_true",
        help="Skip Spark MLlib RandomForest/GBT models (run baselines + XGBoost only).",
    )

    bq_p = subparsers.add_parser(
        "publish-bigquery",
        help="Generate BigQuery schemas/DDL (--dry-run) or upload Gold/Silver tables.",
    )
    bq_p.add_argument("--project", required=True, help="Target GCP project ID.")
    bq_p.add_argument("--dataset", required=True, help="Target BigQuery dataset ID.")
    bq_p.add_argument("--gold-input", default=None, help="Gold tables directory.")
    bq_p.add_argument("--silver-input", default=None, help="Optional Silver path to include.")
    bq_p.add_argument(
        "--table-format",
        choices=["auto", "parquet", "delta"],
        default="auto",
        help="Format for --silver-input.",
    )
    bq_p.add_argument("--tables", nargs="+", default=None, help="Specific Gold tables to publish.")
    bq_p.add_argument("--location", default="US", help="BigQuery dataset location.")
    bq_p.add_argument("--output", default=None, help="Publish manifest JSON path.")
    bq_p.add_argument(
        "--execute",
        action="store_true",
        help="Execute live BigQuery upload using ADC (defaults to safe --dry-run planning).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point returning POSIX exit status codes."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    handlers = {
        "ingest": _handle_ingest,
        "process": _handle_process,
        "gold": _handle_gold,
        "analyze": _handle_analyze,
        "profile": _handle_profile,
        "experiment": _handle_experiment,
        "lakehouse": _handle_lakehouse,
        "benchmark": _handle_benchmark,
        "ingest-soda": _handle_ingest_soda,
        "stream": _handle_stream,
        "validate": _handle_validate,
        "ml": _handle_ml,
        "publish-bigquery": _handle_publish_bigquery,
    }

    try:
        cfg = load_config(config_path=args.config, profile=args.profile)
        return handlers[args.command](args, cfg)
    except Exception:
        logger.exception("Command '%s' failed", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
