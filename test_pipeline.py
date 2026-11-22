"""Offline checks. No network, no API keys, no spend.

Covers the three things that were actually broken: which image the loop returns,
whether RAG failure can take generation down with it, and the retrieval filter.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "app"))
sys.path.insert(0, REPO_ROOT)

from services import dspy_optimization as opt  # noqa: E402


class FakeResult:
    """Stands in for a dspy ChainOfThought result."""

    def __init__(self, score, all_match=False):
        self.overall_score = score
        for f in ("subject_match", "art_type_match", "art_style_match",
                  "art_movement_match", "overall_prompt_match"):
            setattr(self, f, all_match)
        self.has_conflicting_elements = False
        for f in ("subject_feedback", "art_type_feedback", "art_style_feedback",
                  "art_movement_feedback", "overall_prompt_match_feedback",
                  "conflict_description"):
            setattr(self, f, "")
        self.revised_prompt = "revised"


def _service(monkeypatch, scores, all_match_on=None):
    """A service whose 5 iterations yield the given scores. Nothing touches the network."""
    svc = opt.ImageGeneratorService.__new__(opt.ImageGeneratorService)
    svc.image_manager = type("Stub", (), {"add_image": lambda *a, **k: None})()

    calls = {"n": 0}

    def fake_generate(prompt):
        i = calls["n"]
        return f"image-{i}", f"url-{i}", {"url": f"url-{i}"}

    svc.generate_image = fake_generate

    def fake_predictor(**kwargs):
        i = calls["n"]
        calls["n"] += 1
        return FakeResult(scores[i], all_match=(all_match_on == i))

    monkeypatch.setattr(opt.dspy, "ChainOfThought", lambda sig: fake_predictor)
    monkeypatch.setattr(opt, "_retrieve_few_shot_examples", lambda *a, **k: "")
    # Return the URL itself instead of downloading, so we can assert which was picked.
    monkeypatch.setattr(opt, "_download_as_base64_jpeg", lambda url: url)
    return svc


def test_returns_best_scoring_image_not_the_last(monkeypatch):
    """The 15% bug: 11 of 15 non-converged runs returned a worse image than one already made."""
    svc = _service(monkeypatch, scores=[4, 9, 3, 5, 2])
    assert svc.dspy_opt("a fox") == "url-1", "should return the score-9 image, not the last"


def test_converged_run_returns_the_image_that_passed(monkeypatch):
    """When every check passes, that image is the answer regardless of numeric score."""
    svc = _service(monkeypatch, scores=[4, 7, 10, 1, 1], all_match_on=2)
    assert svc.dspy_opt("a fox") == "url-2"


def test_unparseable_score_does_not_crash(monkeypatch):
    """overall_score is LLM output; it may not be an int."""
    svc = _service(monkeypatch, scores=["excellent", None, 6, "n/a", ""])
    assert svc.dspy_opt("a fox") == "url-2"


def test_rag_failure_never_breaks_generation(monkeypatch):
    """RAG is an enhancement. Any failure must degrade to no examples."""
    def explode(*a, **k):
        raise RuntimeError("no UPSTAGE_API_KEY")

    monkeypatch.setattr(opt, "_retriever", None)
    monkeypatch.setitem(sys.modules, "rag.document_processor",
                        type("M", (), {"DocumentProcessor": explode}))
    assert opt._retrieve_few_shot_examples("a fox") == ""


def test_retrieval_filter_selects_successfully_revised_prompts():
    """Only prompts that scored above 8 AND were actually rewritten by the loop."""
    from rag.document_processor import DocumentProcessor

    csv = os.path.join(REPO_ROOT, "app", "imagesdata", "images_log.csv")
    docs = DocumentProcessor.load_successful_prompts(csv)
    assert len(docs) == 34, f"expected the 34 known qualifying prompts, got {len(docs)}"
    assert all(d.page_content.strip() for d in docs)


def test_prompt_template_resolves_regardless_of_cwd(tmp_path, monkeypatch):
    """It used to silently fall back to a leftover 'banner' prompt off-cwd."""
    from models.prompt_manager import PromptManager

    monkeypatch.chdir(tmp_path)
    out = PromptManager.format_prompt("a red fox")
    assert out == "a red fox"
    assert "banner" not in out.lower()
