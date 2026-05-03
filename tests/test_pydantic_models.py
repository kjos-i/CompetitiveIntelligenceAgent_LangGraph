"""Unit tests for the shared Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pydantic_models import Company, SentimentResult


# ---------------------------------------------------------------------------
# Company
# ---------------------------------------------------------------------------

class TestCompany:
    def test_minimal_valid(self):
        c = Company(name="Pepsi")
        assert c.name == "Pepsi"
        assert c.aliases == []
        assert c.special_focus is None

    def test_with_aliases(self):
        c = Company(name="Coca-Cola", aliases=["Coke", "TCCC"])
        assert c.aliases == ["Coke", "TCCC"]

    def test_with_special_focus(self):
        c = Company(name="Pepsi", special_focus="watch Europe launches")
        assert c.special_focus == "watch Europe launches"

    def test_strips_whitespace_on_name(self):
        # str_strip_whitespace is enabled in model_config.
        c = Company(name="  Pepsi  ")
        assert c.name == "Pepsi"

    def test_missing_name_fails(self):
        with pytest.raises(ValidationError):
            Company()  # type: ignore[call-arg]

    def test_aliases_default_empty_list(self):
        c = Company(name="Brand")
        assert isinstance(c.aliases, list)
        assert c.aliases == []


# ---------------------------------------------------------------------------
# SentimentResult
# ---------------------------------------------------------------------------

class TestSentimentResult:
    def test_valid_score(self):
        r = SentimentResult(sentiment="positive", sentiment_score=8)
        assert r.sentiment == "positive"
        assert r.sentiment_score == 8

    def test_score_at_lower_bound(self):
        r = SentimentResult(sentiment="negative", sentiment_score=1)
        assert r.sentiment_score == 1

    def test_score_at_upper_bound(self):
        r = SentimentResult(sentiment="positive", sentiment_score=10)
        assert r.sentiment_score == 10

    def test_score_below_minimum_fails(self):
        with pytest.raises(ValidationError):
            SentimentResult(sentiment="negative", sentiment_score=0)

    def test_score_above_maximum_fails(self):
        with pytest.raises(ValidationError):
            SentimentResult(sentiment="positive", sentiment_score=11)

    def test_sentiment_required(self):
        with pytest.raises(ValidationError):
            SentimentResult(sentiment_score=5)  # type: ignore[call-arg]
