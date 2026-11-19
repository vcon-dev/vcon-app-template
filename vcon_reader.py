"""
Read call details out of stored vCons (vcon-lib 1.x, vCon core-05).

vcon-lib 1.x reads vCons that declare syntax "0.4.0" or "0.5.0"; a 0.4.0 vCon
is migrated to 0.5.0 when it is loaded. Party entries are plain dictionaries.
Transcripts are analysis entries: a "transcript" entry (a text body, or a JSON
body under encoding "json") or a "wtf_transcription" entry whose JSON body is
a WTF document (draft-howe-vcon-wtf-extension).
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

from vcon import Vcon

logger = logging.getLogger(__name__)

TRANSCRIPT_TYPES = ("transcript", "wtf_transcription")
NO_TRANSCRIPT = "No transcript available"


def load_vcon(document: dict) -> Vcon:
    """Build a Vcon from a stored document (for example a MongoDB record).

    Drops the MongoDB `_id` and turns a BSON datetime `created_at` into an
    ISO 8601 string, which is what the vCon format requires.
    """
    data = {k: v for k, v in document.items() if k != "_id"}
    for key in ("created_at", "updated_at"):
        if isinstance(data.get(key), datetime):
            data[key] = data[key].isoformat()
    return Vcon(data)


def _text_from_body(body: Any) -> Optional[str]:
    """Pull transcript text out of a decoded analysis body."""
    if body is None:
        return None
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        transcript = body.get("transcript")
        # WTF: {"transcript": {"text": ...}, "segments": [...]}
        if isinstance(transcript, dict) and isinstance(transcript.get("text"), str):
            return transcript["text"]
        for key in ("transcript", "text", "full_text"):
            if isinstance(body.get(key), str):
                return body[key]
        segments = body.get("segments")
        if isinstance(segments, list):
            texts = [s.get("text") for s in segments if isinstance(s, dict) and s.get("text")]
            if texts:
                return " ".join(texts)
    return json.dumps(body)


def transcript_text(vcon: Vcon) -> str:
    """Return the vCon's transcript text (the last transcript analysis wins)."""
    transcript = NO_TRANSCRIPT
    for analysis in vcon.analysis:
        if analysis.get("type") not in TRANSCRIPT_TYPES:
            continue
        try:
            text = _text_from_body(Vcon.decoded_body(analysis))
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not decode transcript analysis: {e}")
            text = analysis.get("body") if isinstance(analysis.get("body"), str) else None
        if text:
            transcript = text
    return transcript


def _party_tel(parties: list, index: int) -> Optional[str]:
    if index >= len(parties):
        return None
    party = parties[index]
    if isinstance(party, dict):
        return party.get("tel") or party.get("name") or party.get("mailto")
    return getattr(party, "tel", None)


def call_log_entry(document: dict) -> dict:
    """Turn one stored vCon into the diary's call log entry."""
    vcon = load_vcon(document)
    parties = vcon.parties
    return {
        "when": vcon.created_at,
        "to": _party_tel(parties, 0),
        "from": _party_tel(parties, 1),
        "transcript": transcript_text(vcon),
    }
