"""Tests for vcon_reader (vcon-lib 1.x, vCon core-05)."""

from datetime import datetime, timezone

from vcon import Vcon
from vcon.dialog import Dialog
from vcon.party import Party
from vcon.session import Session

from vcon_reader import NO_TRANSCRIPT, call_log_entry, load_vcon, transcript_text

START = "2025-03-01T15:04:05+00:00"


def build_call(transcript_type="transcript", body="Hello, I'd like to order a pizza."):
    """A core-05 call vCon like the conserver stores: one Session, one recording."""
    vcon = Vcon.build_new()
    vcon.add_party(Party(tel="+15551230001"))
    vcon.add_party(Party(tel="+15551230002"))
    session = vcon.add_session(Session(start=START, duration=30.0, parties=[0, 1]))
    vcon.add_dialog(
        Dialog(
            type="recording",
            start=START,
            parties=[0, 1],
            mediatype="audio/wav",
            body="UklGRg",
            encoding="base64url",
        ),
        session=session,
    )
    if transcript_type == "wtf_transcription":
        vcon.add_wtf_transcription_analysis(
            transcript={"text": body, "language": "en", "duration": 30.0, "confidence": 0.9},
            segments=[{"id": 0, "start": 0.0, "end": 3.0, "text": body, "confidence": 0.9}],
            metadata={"created_at": START, "processed_at": START, "provider": "whisper", "model": "base"},
            dialog_index=0,
        )
    elif transcript_type == "transcript":
        vcon.add_analysis(type="transcript", dialog=0, vendor="deepgram", body=body)
    elif transcript_type == "json":
        vcon.add_analysis(
            type="transcript",
            dialog=0,
            vendor="deepgram",
            body={"transcript": body, "confidence": 0.9},
            encoding="json",
            mediatype="application/json",
        )
    assert vcon.is_valid() == (True, [])
    return vcon


def stored(vcon):
    """The document as it comes back from MongoDB."""
    document = vcon.to_dict()
    document["_id"] = "65f0c0ffee"
    return document


def test_text_transcript():
    entry = call_log_entry(stored(build_call()))

    assert entry["to"] == "+15551230001"
    assert entry["from"] == "+15551230002"
    assert entry["transcript"] == "Hello, I'd like to order a pizza."
    datetime.fromisoformat(entry["when"])


def test_json_transcript():
    entry = call_log_entry(stored(build_call("json", "Where is my order?")))

    assert entry["transcript"] == "Where is my order?"


def test_wtf_transcription():
    entry = call_log_entry(stored(build_call("wtf_transcription", "Can I change my booking?")))

    assert entry["transcript"] == "Can I change my booking?"


def test_no_transcript():
    assert call_log_entry(stored(build_call(None)))["transcript"] == NO_TRANSCRIPT


def test_0_4_0_vcon_is_migrated_on_load():
    document = {
        "_id": "65f0c0ffee",
        "vcon": "0.4.0",
        "uuid": "01928e10-193e-8231-b9a2-279e0d16bc46",
        "created_at": START,
        "parties": [{"tel": "+15551230001"}, {"tel": "+15551230002"}],
        "dialog": [
            {
                "type": "recording",
                "start": START,
                "parties": [0, 1],
                "originator": 0,
                "mediatype": "audio/wav",
                "encoding": "base64url",
                "body": "UklGRg",
            }
        ],
        "analysis": [
            {"type": "transcript", "dialog": 0, "vendor": "deepgram", "encoding": "none", "body": "Old call"}
        ],
    }

    vcon = load_vcon(document)
    entry = call_log_entry(document)

    assert vcon.vcon == "0.5.0"
    assert vcon.is_valid() == (True, [])
    assert entry["transcript"] == "Old call"
    assert entry["to"] == "+15551230001"


def test_bson_datetime_created_at():
    document = stored(build_call())
    document["created_at"] = datetime(2025, 3, 1, 15, 4, 5, tzinfo=timezone.utc)

    entry = call_log_entry(document)

    assert datetime.fromisoformat(entry["when"]) == document["created_at"]


def test_transcript_text_ignores_other_analysis():
    vcon = build_call()
    vcon.add_analysis(type="summary", dialog=0, vendor="openai", body="A pizza order")

    assert transcript_text(vcon) == "Hello, I'd like to order a pizza."
