"""Pydantic models shared across the Moodgruppen agent modules."""

from pydantic import BaseModel, ConfigDict, Field


class Company(BaseModel):
    """A competitor entry loaded from watchlist.json."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(description="Canonical company name used as the ledger key.")
    special_focus: str | None = Field(
        default=None,
        description="Optional extra monitoring focus appended to the default search directive.",
    )
    aliases: list[str] = Field(
        default=[],
        description="Alternative names or brand names included in search queries.",
    )


class SentimentResult(BaseModel):
    """Structured LLM output for post-stream sentiment evaluation.

    Produced by a ``with_structured_output`` call after the main agent stream
    completes.  Stored in the ``Description`` (label) and ``Sentiment`` (score)
    columns of the intel ledger.
    """

    sentiment: str = Field(
        description="One-word sentiment label: positive, negative, neutral, or mixed."
    )
    sentiment_score: int = Field(
        ge=1, le=10,
        description="Numeric sentiment rating from 1 (most negative) to 10 (most positive).",
    )
