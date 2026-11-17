import logging
import os
import uuid
from typing import Dict

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from graph.state import GraphState
from graph.utils.node_metrics import NodeTimer

load_dotenv()

logger = logging.getLogger(__name__)


def generate_audio(state: GraphState) -> Dict:
    """
    LangGraph node: Audio Generation.

    Converts the final podcast script to MP3 audio using ElevenLabs TTS.
    Falls back to free gTTS if ElevenLabs fails or quota is exceeded.

    Phase 2 (#10): NodeTimer wraps the full function to record TTS latency
    to Prometheus (podcast_node_duration_seconds{node_name='generate_audio'}).
    This completes 6/6 node coverage for the per-node latency histogram.
    """
    logger.info("--- GENERATE AUDIO NODE ---")

    script = state.get("script", "").strip()
    if not script:
        raise ValueError("No script found in state for audio generation")

    # Ensure output directory exists
    output_dir = "outputs/audio"
    os.makedirs(output_dir, exist_ok=True)

    # Unique filename per run
    output_filename = f"podcast_{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(output_dir, output_filename)

    # Phase 2 (#10): Record generate_audio node latency to Prometheus.
    # Wraps both ElevenLabs and gTTS fallback paths so total TTS time is captured.
    with NodeTimer("generate_audio"):
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise EnvironmentError("ELEVENLABS_API_KEY not found in environment")

            client = ElevenLabs(api_key=api_key)

            # Generate audio stream with ElevenLabs
            audio_stream = client.text_to_speech.convert(
                text=script,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )

            # Write audio chunks to file
            with open(output_path, "wb") as f:
                for chunk in audio_stream:
                    if chunk:
                        f.write(chunk)
            logger.info("Audio saved to: %s (via ElevenLabs)", output_path)

        except Exception as e:
            logger.warning("ElevenLabs TTS failed or quota exceeded: %s. Falling back to free gTTS...", e)
            try:
                from gtts import gTTS
                # Generate audio using free Google Translate TTS API (no keys needed)
                tts = gTTS(text=script, lang='en', tld='com')
                tts.save(output_path)
                logger.info("Audio saved to: %s (via free gTTS fallback)", output_path)
            except Exception as fallback_exc:
                raise RuntimeError(f"Both ElevenLabs and gTTS fallback failed: {fallback_exc}")

    return {"audio_output": output_path}
