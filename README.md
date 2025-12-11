# DEP Agentic Brain

**Digital Identity Protocol (DEP) Brain** is a modular, multi-agent cognitive architecture designed to simulate human-like reasoning. It decomposes complex queries into specialized brain regions, each powered by optimized LLMs, to deliver balanced, deep, and creative responses.

## 🧠 Architecture

The system mimics biological brain functions using 4 distinct agents:

1.  **Prefrontal Cortex (The Executive)**:
    *   **Role**: Planning, Decision Making, & Routing.
    *   **Logic**: Analyzes queries to choose the best processing flow (`Auto`, `Logical`, `Creative`, `Fast`). Breakdowns complex tasks into actionable plans.
    *   **Model**: *Gemini 2.5 Flash-Lite* (for ultra-low latency routing).

2.  **Hippocampus (The Context Provider)**:
    *   **Role**: Memory & Context Retrieval.
    *   **Logic**: Enriches the PFC's plan with relevant historical analogies and concepts. (Currently simulated RAG).

3.  **Left Hemisphere (The Engineer)**:
    *   **Role**: Logic, Structure, & Facts.
    *   **Logic**: Executes the plan with precision, producing dry, verifiable facts. Temperature: `0.0`.

4.  **Right Hemisphere (The Artist)**:
    *   **Role**: Creativity, Synthesis, & Emotion.
    *   **Logic**: Transforms logical data into engaging, human-like narratives. Temperature: `0.9`.

## 🌊 Processing Flows

The brain dynamically chooses how to think based on the query:
*   **Sequential (Deep Thought)**: `PFC -> Hippocampus -> Left -> Right`. (Standard Chain)
*   **Parallel (The Council)**: `Left` and `Right` think simultaneously, then `PFC` synthesizes. (Broad perspective)
*   **Fast (Reflex)**: `PFC` answers immediately. (Zero-latency for greetings/trivial tasks)
*   **Logical**: `PFC -> Hippocampus -> Left`. (For math/code)
*   **Creative**: `PFC -> Hippocampus -> Right`. (For poetry/art)

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Google Gemini API Key

### Installation

1.  Clone the repo.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up environment:
    ```bash
    cp .env.example .env
    # Add your GEMINI_API_KEY in .env
    ```

### Usage

**Auto Mode (Recommended)**: Let the brain decide how to think.
```bash
python3 main.py "Explain gravity"
```

**Force a specific flow**:
```bash
python3 main.py "Hi there" --flow fast
python3 main.py "Write a poem" --flow creative
python3 main.py "Solve X+Y" --flow logical
```
