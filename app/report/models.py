"""Validated, immutable report planning models."""

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ReportTaskRoute: TypeAlias = Literal["rag", "sql"]
ReportDocumentType: TypeAlias = Literal["10-K", "10-Q", "8-K", "20-F", "40-F"]


class ReportRAGScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    company: str | None = Field(default=None, strict=True)
    ticker: str | None = Field(default=None, strict=True)
    fiscal_year: int | None = Field(default=None, ge=1900, le=2100, strict=True)
    document_type: ReportDocumentType | None = None

    @field_validator("company", "ticker", "document_type")
    @classmethod
    def text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("scope text must not be blank")
        return value

    @model_validator(mode="after")
    def must_identify_company(self) -> "ReportRAGScope":
        if self.company is None and self.ticker is None:
            raise ValueError("RAG scope requires company or ticker")
        return self


class ReportTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    task_id: str = Field(max_length=64, pattern=r"^[a-z][a-z0-9_]*$", strict=True)
    question: str = Field(max_length=1000, strict=True)
    expected_route: ReportTaskRoute
    rag_scope: ReportRAGScope | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @model_validator(mode="after")
    def scope_must_match_route(self) -> "ReportTask":
        if self.expected_route == "rag" and self.rag_scope is None:
            raise ValueError("rag tasks require rag_scope")
        if self.expected_route == "sql" and self.rag_scope is not None:
            raise ValueError("sql tasks must not have rag_scope")
        return self


class ReportPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    title: str = Field(max_length=200, strict=True)
    tasks: tuple[ReportTask, ...] = Field(min_length=1, max_length=12)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must not be blank")
        return value

    @model_validator(mode="after")
    def tasks_must_be_unique(self) -> "ReportPlan":
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id values must be unique")

        questions = [task.question.strip().casefold() for task in self.tasks]
        if len(questions) != len(set(questions)):
            raise ValueError(
                "questions must be unique after trimming and case folding"
            )
        return self
