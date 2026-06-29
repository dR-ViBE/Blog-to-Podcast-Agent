from typing import Dict
import os
import uuid
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from graph.state import GraphState

load_dotenv()


import re

def preprocess_for_audio(text: str) -> str:
    """Audio Producer prep: cleans the script for optimal TTS reading."""
    # Remove markdown bold/italic
    text = re.sub(r'[\*\#\_]', '', text)
    # Remove bracketed non-verbal cues e.g. [laughs]
    text = re.sub(r'\[.*?\]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_audio(state: GraphState) -> Dict:
    print("--- GENERATE AUDIO NODE ---")

    script = state.get("script", "").strip()
    if not script:
        raise ValueError("No script found in state for audio generation")
    
    script = preprocess_for_audio(script)

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise EnvironmentError("ELEVENLABS_API_KEY not found in environment")

    client = ElevenLabs(api_key=api_key)

    # Ensure output directory exists
    output_dir = "outputs/audio"
    os.makedirs(output_dir, exist_ok=True)

    # Unique filename per run
    output_filename = f"podcast_{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(output_dir, output_filename)

    # Generate audio stream
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

    print(f"Audio saved to: {output_path}")

    return {
        "audio_output": output_path
    }
