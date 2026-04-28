"""Voice pipeline - Whisper speech-to-text for voice input."""
import tempfile
import whisper

_model = None


def get_whisper_model():
    """Load Whisper model (cached after first call)."""
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model


def transcribe_audio(audio_bytes):
    """Transcribe audio bytes to text using Whisper.

    Args:
        audio_bytes: Raw audio data (WAV format)

    Returns:
        dict with 'text' and 'language'
    """
    model = get_whisper_model()

    # Write audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    result = model.transcribe(temp_path)

    return {
        "text": result["text"].strip(),
        "language": result.get("language", "en"),
    }
