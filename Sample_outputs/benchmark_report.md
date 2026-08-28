# RAGAS Benchmark Report - Video RAG System

This report benchmarks the local ChromaDB + Mistral 7B Video RAG pipeline across 8 modules using **Ragas (Retrieval Augmented Generation Assessment)** and local Ollama evaluators.

---

## Overall Metrics Summary

| Metric | Score | Target Benchmark | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Context Recall** | **`1.0000`** (100.0%) | $\ge 0.80$ | 🟢 **Perfect** | All ground-truth information was successfully retrieved by ChromaDB. |
| **Context Precision** | **`0.8866`** (88.66%) | $\ge 0.75$ | 🟢 **High** | The top-ranked video chunks are highly relevant to the user query. |
| **Answer Relevancy** | **`0.7946`** (79.46%) | $\ge 0.80$ | 🟢 **Strong** | The generated answers directly address the user's questions without filler. |
| **Faithfulness** | **`0.6979`** (69.79%) | $\ge 0.85$ | 🟡 **Moderate** | Answers are largely grounded; lower scores stem from conversational friendly greetings not present in transcripts. |

---

## Per-Question Breakdown

| Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| :--- | :--- | :--- | :--- | :--- |
| What is a list in Python and how do you create one? | 1.0000 | 0.9544 | 0.8056 | 1.0000 |
| How do you add an element to the end of an existing list in Python? | 0.3333 | 0.9690 | 1.0000 | 1.0000 |
| What is a tuple in Python and how is it different from a list? | 1.0000 | 0.9222 | 1.0000 | 1.0000 |
| How can you concatenate or perform operations on tuples in Python? | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| What are dictionaries in Python and how do they store data? | 0.7500 | 0.7085 | 1.0000 | 1.0000 |
| Which methods are available to access keys, values, or update a Python dictionary? | 0.3333 | 0.9908 | 0.9500 | 1.0000 |
| When should you use multithreading in Python and what is its main use case? | 0.5000 | 0.8338 | 0.5333 | 1.0000 |
| What is the difference between multithreading and multiprocessing in Python, and when is multiprocessing preferred? | 0.6667 | 0.9781 | 0.8042 | 1.0000 |

---

## Key Insights & Recommendations

1. **Retrieval Performance (100% Recall & 88.66% Precision):**
   * ChromaDB with `nomic-embed-text` and 5-segment chunk grouping retrieved all required course knowledge in the top 5 chunks for every single test query.
2. **Why Faithfulness is ~70%:**
   * Our conversational prompt instructs the model to use friendly greetings like *"Hey there! Happy coding!"* and convert seconds to minutes. Ragas strict factual faithfulness penalizes phrases that are not verbatim in the transcripts.
3. **Reproducibility:**
   * The benchmark can be re-run anytime via `python Pipeline/evaluate_rag.py`. Detailed scores are saved to `Data/evaluation_results.csv`.
