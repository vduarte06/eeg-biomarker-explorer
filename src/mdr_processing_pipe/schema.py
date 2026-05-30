"""Pydantic schema for the pipeline YAML.

Validates the base structure — required fields, types, and structural integrity.
Step-level config (filter frequencies, ICA components, etc.) is intentionally
left unvalidated here; errors there surface naturally at execution time.
"""

from typing import Any
from pydantic import BaseModel, field_validator, model_validator


class DatasetConfig(BaseModel):
    path: str
    session_log: str | None = None


class EventConfig(BaseModel):
    annotation: str


class InputConfig(BaseModel):
    dataset: DatasetConfig
    events:  dict[str, EventConfig] = {}
    regions: dict[str, list[str]]   = {}


class PipelineStep(BaseModel):
    """A single pipeline step: a dict with exactly one key (the step type)."""
    root: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def exactly_one_key(cls, v: Any) -> Any:
        if not isinstance(v, dict) or len(v) != 1:
            raise ValueError(
                f"Each pipeline step must be a mapping with exactly one key, got: {v!r}"
            )
        return {"root": v}

    @property
    def kind(self) -> str:
        return next(iter(self.root))

    @property
    def config(self) -> dict:
        return self.root[self.kind] or {}


class PipelineConfig(BaseModel):
    preprocessing:       list[PipelineStep] = []
    feature_extraction:  list[PipelineStep] = []

    @field_validator("preprocessing", "feature_extraction", mode="before")
    @classmethod
    def coerce_steps(cls, v: Any) -> list:
        return v or []


class OutputConfig(BaseModel):
    format: list[str] = ["csv"]
    path:   str       = "./data/processed"

    @field_validator("format")
    @classmethod
    def supported_formats(cls, v: list[str]) -> list[str]:
        allowed = {"csv", "parquet"}
        bad = set(v) - allowed
        if bad:
            raise ValueError(f"Unsupported output format(s): {bad}. Allowed: {allowed}")
        return v


class PipelineSchema(BaseModel):
    version:  int          = 1
    input:    InputConfig
    pipeline: PipelineConfig = PipelineConfig()
    output:   OutputConfig    = OutputConfig()
