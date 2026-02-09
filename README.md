# 🎙️ Blog-to-Podcast Agent

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Flow-orange?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local_Embeddings-black?style=for-the-badge&logo=ollama)

**Blog-to-Podcast Agent** is an intelligent workflow that converts technical blog posts into engaging audio podcasts. 

Unlike simple text-to-speech tools, this project uses an **agentic workflow (LangGraph)** to ingest content, generate a conversational script, grade the quality of the script, and iteratively improve it before generating the final audio using **ElevenLabs**.

---

## 🏗️ Agent Architecture

This project uses a cyclic graph to ensure high-quality script generation.

```mermaid
graph TD
    Start([Start]) --> Ingest[Ingest & Embed Blog]
    Ingest --> Retrieve[Retrieve Context]
    Retrieve --> GenScript[Generate Podcast Script]
    GenScript --> Grade{Grade Script}
    
    Grade -->|Needs Improvement| Suggest[Suggest Improvements]
    Suggest --> GenScript
    
    Grade -->|Approved| GenAudio[Generate Audio (ElevenLabs)]
    GenAudio --> End([End])
    
    style Start fill:#f9f,stroke:#333
    style GenScript fill:#bbf,stroke:#333
    style Grade fill:#ff9,stroke:#333
    style GenAudio fill:#9f9,stroke:#333
