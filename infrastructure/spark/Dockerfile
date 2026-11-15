FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    SPARK_EXTRA_JARS_DIR=/opt/spark-extra-jars

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      openjdk-17-jre-headless \
      curl \
      procps \
      tini && \
    ln -s /usr/lib/jvm/java-17-openjdk-* /usr/lib/jvm/java-default && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-default
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
RUN uv export --frozen --no-dev --no-emit-project --output-file /tmp/requirements.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Pre-download pinned Delta Lake, Hadoop S3A, and Kafka Streaming JARs so containers never need Maven at runtime.
RUN mkdir -p /opt/spark-extra-jars && \
    curl -fsSL -o /opt/spark-extra-jars/delta-spark_2.12-3.2.0.jar \
      https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/3.2.0/delta-spark_2.12-3.2.0.jar && \
    curl -fsSL -o /opt/spark-extra-jars/delta-storage-3.2.0.jar \
      https://repo1.maven.org/maven2/io/delta/delta-storage/3.2.0/delta-storage-3.2.0.jar && \
    curl -fsSL -o /opt/spark-extra-jars/hadoop-aws-3.3.4.jar \
      https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -fsSL -o /opt/spark-extra-jars/aws-java-sdk-bundle-1.12.262.jar \
      https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar && \
    curl -fsSL -o /opt/spark-extra-jars/spark-sql-kafka-0-10_2.12-3.5.1.jar \
      https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.1/spark-sql-kafka-0-10_2.12-3.5.1.jar && \
    curl -fsSL -o /opt/spark-extra-jars/spark-token-provider-kafka-0-10_2.12-3.5.1.jar \
      https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.1/spark-token-provider-kafka-0-10_2.12-3.5.1.jar && \
    curl -fsSL -o /opt/spark-extra-jars/kafka-clients-3.4.1.jar \
      https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar && \
    curl -fsSL -o /opt/spark-extra-jars/commons-pool2-2.11.1.jar \
      https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar && \
    SPARK_JARS_DIR=$(python -c "import pyspark, pathlib; print(pathlib.Path(pyspark.__file__).parent / 'jars')") && \
    cp /opt/spark-extra-jars/*.jar "${SPARK_JARS_DIR}/"

COPY src /app/src
COPY configs /app/configs
COPY infrastructure/spark/entrypoint.sh /usr/local/bin/spark-entrypoint.sh

RUN chmod +x /usr/local/bin/spark-entrypoint.sh && \
    sed -i 's/\r$//' /usr/local/bin/spark-entrypoint.sh && \
    uv pip install --system --no-cache --no-deps .

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/spark-entrypoint.sh"]
CMD ["python", "-m", "chicago_crimes", "--help"]
