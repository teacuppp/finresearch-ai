"""Strict, cited claims produced by report synthesis."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EvidenceID = Annotated[str, Field(pattern=r"^E[1-9][0-9]*$", strict=True)]


class ReportClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    task_id: str = Field(max_length=64, pattern=r"^[a-z][a-z0-9_]*$", strict=True)
    text: str = Field(max_length=1500, strict=True)
    evidence_ids: tuple[EvidenceID, ...] = Field(min_length=1, max_length=12, strict=True)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim text must not be blank")
        return value

    @model_validator(mode="after")
    def evidence_ids_must_be_unique(self) -> "ReportClaim":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("claim evidence IDs must be unique")
        return self


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    heading: str = Field(max_length=120, strict=True)
    claims: tuple[ReportClaim, ...] = Field(min_length=1, max_length=20, strict=True)

    @field_validator("heading")
    @classmethod
    def heading_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("section heading must not be blank")
        return value


class ReportDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    title: str = Field(max_length=200, strict=True)
    sections: tuple[ReportSection, ...] = Field(min_length=1, max_length=12, strict=True)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("report title must not be blank")
        return value
