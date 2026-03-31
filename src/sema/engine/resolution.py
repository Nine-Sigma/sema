"""REMOVED: ResolutionEngine has been retired.

All assertion-to-graph logic is now consolidated in the unified materializer:
    from sema.graph.materializer import materialize_unified

See docs/architecture/warehouse-profiling-and-feedback-loop.md
and openspec/changes/pipeline-consolidation-v2/ for details.
"""

from __future__ import annotations


def __getattr__(name: str) -> None:
    if name == "ResolutionEngine":
        raise ImportError(
            "ResolutionEngine has been removed. "
            "Use sema.graph.materializer.materialize_unified instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
