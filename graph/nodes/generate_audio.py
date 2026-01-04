from typing import Dict
from graph.state import GraphState


def generate_audio(state: GraphState) -> Dict:
    print("GENERATE AUDIO NODE")
    # script = state.get("script", "")
    audio = "placeholder"
    return {"audio_output": audio}
