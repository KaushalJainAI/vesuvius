"""Claude Agent SDK parallel pathway.

Runs alongside the OpenAI / local agent pipeline so that the same segments
can be read by Claude (via the Claude Code subscription that the user is
already authenticated against). Output schema matches `rich_reading.py` so
the webapp can render the two side-by-side.

Behaviour:
  - If `claude_agent_sdk` is importable AND `claude` CLI is on PATH (or
    `ANTHROPIC_API_KEY` is set), it runs the SDK against each strip image.
  - Otherwise it writes a graceful "claude path not available" stub so the
    webapp comparison panel still has something to render.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
PRED_ROOT = ROOT / "predictions" / "decipher"
WEB_ROOT = ROOT / "web" / "public" / "assets" / "decipher"
AGENT_ROOT = ROOT / "agent_decoding"


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

def claude_available() -> Dict[str, Any]:
    """Detect whether the Claude SDK path can actually run."""
    info: Dict[str, Any] = {"sdk_import": False, "cli_on_path": False, "api_key": False}
    try:
        import claude_agent_sdk  # noqa: F401
        info["sdk_import"] = True
        info["sdk_version"] = getattr(claude_agent_sdk, "__version__", "unknown")
    except Exception as e:
        info["sdk_error"] = repr(e)
        return info
    info["cli_on_path"] = shutil.which("claude") is not None
    info["api_key"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    info["ok"] = info["sdk_import"] and (info["cli_on_path"] or info["api_key"])
    return info


# ---------------------------------------------------------------------------
# Prompt construction (re-uses the rich_reading schema)
# ---------------------------------------------------------------------------

_STRIP_PROMPT = """You are a classical-Greek paleographer reading a heavily-damaged
CT-derived ink probability map of a Herculaneum scroll strip. You also receive
the existing ensemble consensus and template-match hints. Produce ONE raw JSON
object (no markdown fences) with the schema:

{
  "letters_text": "...",                # joined Greek uncials in reading order
  "transcription_lines": [               # one entry per visible letter position
    {"index": 0, "char": "Ρ", "alternates": ["Ϲ"], "confidence": 0.5,
     "tier": "MED", "bbox_strip": [x,y,w,h], "word_index": 0, "is_word_boundary": false}
  ],
  "word_candidates": [
    {"word_index": 0, "greek_letters": "ΡΟΤΕΩΡ",
     "matches": [{"greek": "ΛΟΓΟΣ", "english": "reason", "score": 0.6}]}
  ],
  "paraphrase_en": "1-2 sentence probable meaning",
  "interpretation": "1 paragraph weaving recognised words into a candidate reading",
  "caveats": "honest one-line uncertainty note"
}

Anchor everything to what you can actually see in the image; deviate from the
consensus only when the shapes clearly demand it. Empty letters[] is acceptable
if the strip is unreadable. Use proper Greek uncial chars (ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ, lunate Ϲ)."""


def _strip_user_message(seg_id: str, strip: Dict[str, Any], strip_path: Path) -> str:
    """Compose the per-strip Claude-SDK prompt as a single text message.

    Note: the Claude Agent SDK's text-only `query()` doesn't take raw image
    bytes - it expects a file path the agent can open with the Read tool.
    We hand it the absolute strip PNG path; the agent calls Read internally.
    """
    sid = int(strip.get("strip_id", 0))
    consensus = (strip.get("consensus") or {})
    consensus_lite = {
        "text": consensus.get("text", ""),
        "characters": [
            {
                "char": c.get("char"),
                "x_norm": c.get("x_norm"),
                "confidence": c.get("confidence"),
                "alternatives": c.get("alternatives", []),
                "bbox_strip": c.get("bbox_strip"),
            }
            for c in (consensus.get("characters") or [])
        ],
    }
    hints = [
        {"x_norm": h.get("x_norm"),
         "candidates": [(c[0], round(float(c[1]), 3))
                        for c in (h.get("candidates") or [])[:3]]}
        for h in (strip.get("template_hints") or [])
    ]
    return (
        f"{_STRIP_PROMPT}\n\n"
        f"Segment: {seg_id}  Strip: {sid}\n"
        f"Open the image at this absolute path with the Read tool: {strip_path}\n\n"
        f"## Ensemble consensus\n```json\n{json.dumps(consensus_lite, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## Template hints\n```json\n{json.dumps(hints, ensure_ascii=False, indent=2)}\n```\n\n"
        f"Now look at the image and produce the JSON object."
    )


# ---------------------------------------------------------------------------
# Run one strip through the Claude Agent SDK
# ---------------------------------------------------------------------------

async def _claude_one_strip(seg_id: str, strip: Dict[str, Any], strip_path: Path) -> Dict[str, Any]:
    from claude_agent_sdk import query, ClaudeAgentOptions

    prompt = _strip_user_message(seg_id, strip, strip_path)
    options = ClaudeAgentOptions(
        system_prompt="You are a careful classical-Greek paleographer. Return ONLY JSON.",
        allowed_tools=["Read"],
        max_turns=4,
    )

    text_chunks: List[str] = []
    async for msg in query(prompt=prompt, options=options):
        # The SDK streams AssistantMessage with TextBlock children.
        try:
            from claude_agent_sdk import AssistantMessage, TextBlock
        except Exception:
            AssistantMessage = TextBlock = None  # type: ignore[assignment]
        if AssistantMessage is not None and isinstance(msg, AssistantMessage):
            for block in getattr(msg, "content", []) or []:
                if TextBlock is not None and isinstance(block, TextBlock):
                    text_chunks.append(block.text or "")

    raw = "\n".join(text_chunks).strip()
    return _parse_strip_json(raw)


def _parse_strip_json(raw: str) -> Dict[str, Any]:
    """Tolerant JSON parser - strip code fences, extract first {...} block."""
    if not raw:
        return {"_error": "empty model output"}
    s = raw.strip()
    if s.startswith("```"):
        # Strip leading and trailing code fence
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        s = s.rstrip("` \n\t")
    if not s.startswith("{"):
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            s = s[i:j + 1]
    try:
        return json.loads(s)
    except Exception as e:
        return {"_error": f"json parse: {e!r}", "raw": raw[:600]}


# ---------------------------------------------------------------------------
# Public entry: run for one segment
# ---------------------------------------------------------------------------

async def run_segment_async(seg_id: str) -> Dict[str, Any]:
    seg_pred = PRED_ROOT / seg_id
    result_path = seg_pred / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"no result.json for {seg_id}")
    result = json.loads(result_path.read_text(encoding="utf-8"))

    avail = claude_available()
    out_dir = AGENT_ROOT / seg_id
    out_dir.mkdir(parents=True, exist_ok=True)
    web_dir = WEB_ROOT / seg_id / "agent"
    web_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "seg_id": seg_id,
        "source": "claude-agent-sdk",
        "availability": avail,
        "strips": [],
    }

    if not avail.get("ok"):
        # Graceful stub so the webapp panel still renders.
        payload["status"] = "unavailable"
        payload["message"] = (
            "Claude Agent SDK is installed but not runnable here: "
            f"sdk_import={avail.get('sdk_import')} cli_on_path={avail.get('cli_on_path')} "
            f"api_key={avail.get('api_key')}. Set ANTHROPIC_API_KEY or ensure the "
            "`claude` CLI is on PATH to enable this pathway."
        )
        (out_dir / "claude_reading.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (web_dir / "claude_reading.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[claude] {seg_id}: {payload['status']} - {payload['message'][:80]}")
        return payload

    payload["status"] = "ok"
    for strip in result.get("strips") or []:
        sid = int(strip.get("strip_id", 0))
        strip_path = (seg_pred / strip.get("image_path", f"strips/strip_{sid:02d}.png")).resolve()
        if not strip_path.exists():
            continue
        print(f"  - claude strip {sid} ...")
        try:
            parsed = await _claude_one_strip(seg_id, strip, strip_path)
            parsed["strip_id"] = sid
            payload["strips"].append(parsed)
        except Exception as e:
            payload["strips"].append({"strip_id": sid, "_error": repr(e)})

    (out_dir / "claude_reading.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (web_dir / "claude_reading.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_segment_sync(seg_id: str) -> Dict[str, Any]:
    return asyncio.run(run_segment_async(seg_id))
