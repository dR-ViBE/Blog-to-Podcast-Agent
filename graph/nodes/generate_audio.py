from graph.state import GraphState


def generate_audio(state: GraphState):
    script = state.get("script", "")
    audio = "placeholder"
    return {"audio_output": audio}
