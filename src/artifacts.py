"""Artifact types for the pipeline.

Artifacts represent non-text outputs (charts, images, data files) produced
by agents or tools. They flow through PipelineState and can be embedded
as multimodal content in LLM messages.

Usage:
    from src.artifacts import Artifact, ArtifactType

    artifact = Artifact(
        artifact_type=ArtifactType.IMAGE,
        path="outputs/AAPL_sentiment.png",
        mime_type="image/png",
        description="AAPL sentiment over time",
    )
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactType(str, Enum):
    """Types of non-text outputs."""

    IMAGE = "image"  # PNG, JPG, SVG
    FILE = "file"  # CSV, PDF, etc.
    DATA = "data"  # In-memory structured data


@dataclass
class Artifact:
    """A non-text output produced by an agent or tool.

    Carries a file path, type, and metadata. Provides helpers to convert
    images to base64 and LangChain multimodal content blocks.
    """

    artifact_type: ArtifactType
    path: str
    mime_type: str = "application/octet-stream"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        """Check if this artifact is an image."""
        return self.artifact_type == ArtifactType.IMAGE

    def to_base64(self) -> str:
        """Read the file and return base64-encoded content."""
        data = Path(self.path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def to_multimodal_block(self) -> dict:
        """Convert to a LangChain-compatible image_url content block."""
        b64 = self.to_base64()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{self.mime_type};base64,{b64}"},
        }
