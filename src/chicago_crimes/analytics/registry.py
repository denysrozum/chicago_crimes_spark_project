"""Central registry mapping all 24 analytical tasks across legacy and domain identifiers."""

from __future__ import annotations

from chicago_crimes.analytics.arrests import ARREST_TASKS
from chicago_crimes.analytics.base import AnalyticalTask
from chicago_crimes.analytics.categories import CATEGORY_TASKS
from chicago_crimes.analytics.geographic import GEOGRAPHIC_TASKS
from chicago_crimes.analytics.temporal import TEMPORAL_TASKS

ALL_TASKS: tuple[AnalyticalTask, ...] = (
    TEMPORAL_TASKS + GEOGRAPHIC_TASKS + CATEGORY_TASKS + ARREST_TASKS
)

ANALYTICAL_TASKS: dict[str, AnalyticalTask] = {task.task_id: task for task in ALL_TASKS}

_ALIAS_INDEX: dict[str, AnalyticalTask] = {}
for _task in ALL_TASKS:
    _ALIAS_INDEX[_task.task_id.lower()] = _task
    _ALIAS_INDEX[_task.legacy_name.lower()] = _task
    _ALIAS_INDEX[_task.domain_name.lower()] = _task


def get_task(identifier: str) -> AnalyticalTask:
    """Resolve an analytical task by task_id, legacy_name, or domain_name."""
    key = identifier.strip().lower()
    if key not in _ALIAS_INDEX:
        valid = ", ".join(t.task_id for t in ALL_TASKS)
        raise KeyError(f"Unknown analytical task '{identifier}'. Available task IDs: {valid}")
    return _ALIAS_INDEX[key]


def resolve_tasks(
    identifiers: list[str] | None = None,
    category: str | None = None,
) -> list[AnalyticalTask]:
    """Return ordered list of analytical tasks filtered by identifiers or category."""
    if identifiers:
        seen: set[str] = set()
        selected: list[AnalyticalTask] = []
        for ident in identifiers:
            task = get_task(ident)
            if task.task_id not in seen:
                seen.add(task.task_id)
                selected.append(task)
        return selected

    if category:
        cat = category.strip().lower()
        matches = [t for t in ALL_TASKS if t.category.lower() == cat]
        if not matches:
            raise ValueError(f"No analytical tasks found for category '{category}'.")
        return matches

    return list(ALL_TASKS)
