"""Pipeline orchestration: claims -> candidates -> judge -> verify -> flags.

Shared by the CLI (scripts/run_pipeline.py) and the web console
(adonis.web.app) so both exercise exactly the same M4 logic.
"""

from __future__ import annotations

from adonis.pipeline.core import PipelineStats, run, wipe_pipeline_rows

__all__ = ["PipelineStats", "run", "wipe_pipeline_rows"]