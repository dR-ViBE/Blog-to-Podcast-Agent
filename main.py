from dotenv import load_dotenv
load_dotenv()
from graph.graph import app
from langchain_core.documents import Document
import os  # Optional, just for debugging



# 1. LOAD ENV VARS FIRST (CRITICAL STEP)


# 2. THEN IMPORT YOUR GRAPH
# Now when this imports, the keys are already available

if __name__ == "__main__":
    print("--- 🚀 STARTING AGENT RUNTIME DRIVER ---")

    # Optional Debug: Check if key is actually loaded
    # print(f"API KEY CHECK: {os.getenv('GROQ_API_KEY')}")

    # ... rest of your code ...
    dummy_blog_content = """
    The Future of AI Agents in 2025.
    Artificial Intelligence is moving beyond simple chatbots.
    """

    test_doc = Document(
        page_content=dummy_blog_content,
        metadata={"source": "manual_test", "title": "AI Future"}
    )

    initial_state = {
        "documents": [test_doc],
        "generation_count": 0,
        "max_generations": 3
    }

    final_state = app.invoke(initial_state)

    print("\n--- 📊 FINAL STATE DUMP ---")
    print(final_state)
    print("\n--- 📝 GENERATED SCRIPT ---")
    print(final_state.get("script", "NO SCRIPT GENERATED"))
