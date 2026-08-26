from __future__ import annotations

import cv2
import numpy as np
import pytest
import sticker.agent as agent_module

from sticker.agent import (
    AgentCall,
    StickerAgentHarness,
    _assessment_prompt,
    _extract_json,
    _extract_json_with_metadata,
    _merge_character_recognition,
    _sanitize_assessment,
    _validate_adjudicator_payload,
    _validate_assessment_payload,
)
from sticker.evidence import analyze_plate_evidence


def _patched_plate() -> np.ndarray:
    image = np.full((290, 992, 3), (125, 220, 160), dtype=np.uint8)
    cv2.rectangle(image, (16, 5), (975, 284), (220, 235, 220), 5)
    for x, character in zip((90, 195, 385, 495, 605, 715, 825, 925), "AZ6Q268B", strict=True):
        cv2.putText(image, character, (x - 36, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.35, (20, 28, 22), 10, cv2.LINE_AA)
    cv2.rectangle(image, (330, 35), (440, 255), (113, 204, 149), -1)
    cv2.rectangle(image, (330, 35), (440, 255), (238, 246, 241), 3)
    cv2.line(image, (334, 252), (438, 252), (28, 44, 32), 3)
    return image


def test_extract_json_accepts_fenced_response() -> None:
    assert _extract_json('```json\n{"decision":"clear"}\n```') == {"decision": "clear"}


@pytest.mark.parametrize(
    "raw",
    [
        '{"decision":"clear" "selected_candidates":[]}',
        '{"decision":"clear","selected_candidates":[],}',
        '{"decision":"clear","selected_candidates":[]',
    ],
)
def test_extract_json_repairs_common_syntax_damage(raw: str) -> None:
    payload, repaired = _extract_json_with_metadata(raw)
    assert payload == {"decision": "clear", "selected_candidates": []}
    assert repaired


def test_local_qwen_request_can_disable_thinking_and_limit_output(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": '{"decision":"clear"}'})()
            choice = type("Choice", (), {"message": message})()
            usage = type("Usage", (), {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})()
            return type("Response", (), {"choices": [choice], "usage": usage})()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("STICKER_AGENT_MODEL", "qwen35-2b-local")
    monkeypatch.setenv("STICKER_AGENT_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("STICKER_AGENT_API_KEY", "EMPTY")
    monkeypatch.setenv("STICKER_AGENT_DISABLE_THINKING", "1")
    monkeypatch.setenv("STICKER_AGENT_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("STICKER_AGENT_ADJUDICATOR_MAX_OUTPUT_TOKENS", "1536")
    harness = StickerAgentHarness(cache_dir=tmp_path, use_cache=False)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert harness._call("test", "system", "prompt", [image]) == {"decision": "clear"}
    assert captured["model"] == "qwen35-2b-local"
    assert captured["max_tokens"] == 4096
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert harness._max_tokens_for_stage("adjudicator") == 1536


def test_local_qwen_request_uses_strict_json_schema(monkeypatch, tmp_path) -> None:
    captured: dict = {}
    payload = {
        "plate_quality": "assessable",
        "candidates": [
            {
                "candidate_id": "C1",
                "slot_id": "S1",
                "observed_character": "A",
                "verdict": "uncertain",
                "suspected_tamper_types": [],
                "possible_originals": [],
                "stroke_regions": [],
                "geometry_observations": [],
                "appearance_observations": [],
                "normal_explanations": [],
                "needs_recheck": False,
            }
        ],
        "summary": "ok",
    }

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": __import__("json").dumps(payload)})()
            choice = type("Choice", (), {"message": message})()
            usage = type("Usage", (), {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})()
            return type("Response", (), {"choices": [choice], "usage": usage})()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("STICKER_AGENT_MODEL", "qwen35-2b-local")
    monkeypatch.setenv("STICKER_AGENT_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("STICKER_AGENT_API_KEY", "EMPTY")
    monkeypatch.setenv("STICKER_AGENT_STRUCTURED_OUTPUT", "1")
    harness = StickerAgentHarness(cache_dir=tmp_path, use_cache=False)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    actual = harness._call("investigator", "system", "prompt", [image], {"C1"}, {"C1": 1})
    assert actual == payload
    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    assert harness.calls[0].structured_output is True
    assert harness.calls[0].json_repaired is False


def test_assessment_validation_rejects_wrong_slot_and_incomplete_coverage() -> None:
    payload = {
        "plate_quality": "assessable",
        "candidates": [
            {
                "candidate_id": "C1",
                "slot_id": "S2",
                "observed_character": None,
                "verdict": "uncertain",
                "suspected_tamper_types": [],
                "possible_originals": [],
                "stroke_regions": [],
                "geometry_observations": [],
                "appearance_observations": [],
                "normal_explanations": [],
                "needs_recheck": False,
            }
        ],
        "summary": "",
    }
    with pytest.raises(ValueError, match="slot_id"):
        _validate_assessment_payload(payload, {"C1", "C2"}, {"C1": 1, "C2": 2})


def test_adjudicator_validation_rejects_overlap_and_non_character() -> None:
    base = {
        "decision": "suspicious",
        "selected_candidates": ["C1"],
        "uncertain_candidates": ["C1"],
        "recognized_characters": {"C1": "看不清"},
        "candidate_evidence": {},
        "reasoning_summary": "",
        "unassessable_reason": None,
    }
    with pytest.raises(ValueError, match="冲突"):
        _validate_adjudicator_payload(base, {"C1"})
    base["uncertain_candidates"] = []
    with pytest.raises(ValueError, match="单个车牌字符"):
        _validate_adjudicator_payload(base, {"C1"})


def test_assessment_drops_unknown_candidate_ids() -> None:
    payload = {
        "candidates": [
            {"candidate_id": "C1", "slot_id": "S99", "verdict": "tamper_support"},
            {"candidate_id": "C99", "verdict": "tamper_support"},
        ]
    }
    actual = _sanitize_assessment(payload, {"C1", "C2"})
    assert [item["candidate_id"] for item in actual["candidates"]] == ["C1"]
    assert actual["candidates"][0]["slot_id"] == "S1"


def test_legacy_or_ambiguous_verdict_is_not_treated_as_tamper_support() -> None:
    actual = _sanitize_assessment(
        {"candidates": [{"candidate_id": "C1", "verdict": "support"}]},
        {"C1"},
    )
    assert actual["candidates"][0]["verdict"] == "uncertain"


def test_character_recognition_uses_consensus_and_rejects_explanatory_text() -> None:
    decision = {
        "selected_candidates": ["C3"],
        "uncertain_candidates": ["C4"],
        "recognized_characters": {"C3": None, "C4": "看不清"},
    }
    _merge_character_recognition(
        decision,
        [
            {"candidates": [{"candidate_id": "C3", "observed_character": "a"}]},
            {"candidates": [{"candidate_id": "C3", "observed_character": "A"}]},
        ],
    )
    assert decision["recognized_characters"] == {"C3": "A", "C4": None}
    assert decision["character_recognition_status"] == {
        "C3": "cross_review_consensus",
        "C4": "unreadable_or_not_returned",
    }


def test_prompt_binds_candidate_to_left_to_right_slot_without_line_count() -> None:
    artifacts = analyze_plate_evidence(_patched_plate())
    prompt = _assessment_prompt(artifacts)
    assert "C1=S1=从左第1位" in prompt
    assert "C8=S8=从左第8位" in prompt
    assert "line_count" not in prompt
    assert "anomaly_score 越高表示越异常" in prompt


def test_final_decision_cannot_invent_coordinates_or_candidate() -> None:
    image = _patched_plate()
    artifacts = analyze_plate_evidence(image)
    actual = StickerAgentHarness._sanitize_final(
        {
            "decision": "suspicious",
            "selected_candidates": ["C3", "C99"],
            "uncertain_candidates": ["C4", "C99"],
            "recognized_characters": {"C3": "A", "C4": "7", "C99": "X"},
            "candidate_evidence": {"C3": {"geometry": ["rectangle"]}},
        },
        {f"C{i}" for i in range(1, 9)},
        artifacts,
    )
    assert actual["selected_candidates"] == ["C3"]
    assert actual["uncertain_candidates"] == []
    assert "C99" not in actual["candidate_evidence"]
    assert actual["recognized_characters"] == {"C3": "A"}


def test_final_decision_suppresses_weak_adjacent_spillover() -> None:
    image = _patched_plate()
    artifacts = analyze_plate_evidence(image)
    actual = StickerAgentHarness._sanitize_final(
        {
            "decision": "suspicious",
            "selected_candidates": ["C3"],
            "uncertain_candidates": ["C4"],
        },
        {f"C{i}" for i in range(1, 9)},
        artifacts,
    )
    assert actual["selected_candidates"] == ["C3"]
    assert actual["uncertain_candidates"] == []


def test_final_decision_does_not_turn_evidence_uncertainty_into_unassessable() -> None:
    image = _patched_plate()
    artifacts = analyze_plate_evidence(image)
    actual = StickerAgentHarness._sanitize_final(
        {
            "decision": "clear",
            "selected_candidates": [],
            "uncertain_candidates": ["C2"],
        },
        {f"C{i}" for i in range(1, 9)},
        artifacts,
    )
    assert actual["decision"] == "suspicious"
    assert actual["selected_candidates"] == ["C3"]
    assert actual["uncertain_candidates"] == []


def test_model_cannot_refuse_assessable_plate() -> None:
    artifacts = analyze_plate_evidence(_patched_plate())
    actual = StickerAgentHarness._sanitize_final(
        {
            "decision": "unassessable",
            "selected_candidates": [],
            "uncertain_candidates": [],
            "unassessable_reason": "证据冲突",
        },
        {f"C{i}" for i in range(1, 9)},
        artifacts,
    )
    assert artifacts.bundle.quality.assessable
    assert actual["decision"] == "suspicious"
    assert actual["selected_candidates"] == ["C3"]
    assert actual["unassessable_reason"] is None


def test_model_cannot_omit_deterministic_candidate() -> None:
    artifacts = analyze_plate_evidence(_patched_plate())
    actual = StickerAgentHarness._sanitize_final(
        {
            "decision": "clear",
            "selected_candidates": [],
            "uncertain_candidates": [],
        },
        {f"C{i}" for i in range(1, 9)},
        artifacts,
    )
    assert actual["decision"] == "suspicious"
    assert actual["selected_candidates"] == ["C3"]
    assert actual["candidate_evidence"]["C3"]["selection_sources"] == [
        "deterministic_multichannel"
    ]


@pytest.mark.parametrize(
    ("max_calls", "expected_stages"),
    [
        (2, ["investigator", "adjudicator"]),
        (3, ["investigator", "reviewer", "adjudicator"]),
        (4, ["investigator", "reviewer", "recheck", "adjudicator"]),
    ],
)
def test_agent_call_budget_controls_review_depth(max_calls: int, expected_stages: list[str]) -> None:
    image = np.full((290, 992, 3), (125, 220, 160), dtype=np.uint8)
    artifacts = analyze_plate_evidence(image)
    harness = object.__new__(StickerAgentHarness)
    harness.model = "fake-model"
    harness.base_url = "https://example.test/v1"
    harness.max_calls_per_image = max_calls
    harness.decision_profile = "balanced"
    harness.calls = []

    def fake_call(
        stage: str,
        system: str,
        prompt: str,
        images: list[np.ndarray],
        allowed_ids: set[str] | None = None,
        slot_by_candidate: dict[str, int] | None = None,
    ) -> dict:
        del system, prompt, images, allowed_ids, slot_by_candidate
        harness.calls.append(AgentCall(stage, stage, False, "", {}, {"input_tokens": 10, "output_tokens": 2}))
        if stage == "investigator":
            return {"candidates": [{"candidate_id": "C1", "verdict": "tamper_support"}]}
        if stage == "reviewer":
            return {"candidates": [{"candidate_id": "C1", "verdict": "normal_support"}]}
        if stage == "recheck":
            return {"candidates": [{"candidate_id": "C1", "verdict": "normal_support"}]}
        return {"decision": "clear", "selected_candidates": [], "uncertain_candidates": []}

    harness._call = fake_call
    _, trajectory = harness.run(image, artifacts)
    assert [call["stage"] for call in trajectory["calls"]] == expected_stages
    assert trajectory["max_calls_per_image"] == max_calls
