Local LLM Based Video RAG System

# Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline to perform semantic question answering over video content.
The system converts video files into searchable knowledge by transcribing audio, generating embeddings for transcript chunks, retrieving relevant content using cosine similarity, and generating answers using a local large language model.
The entire pipeline runs locally using Ollama models.



# Tech Stack

* ffmpeg – Video to MP3 conversion
* Whisper – Speech-to-text transcription
* Ollama (nomic-embed) – Embedding generation
* Pandas – In-memory embedding storage
* sklearn.metrics – Cosine similarity computation
* Mistral (Ollama) – Answer generation



# Pipeline

1. Convert video to MP3 using ffmpeg
2. Transcribe MP3 to text using Whisper
3. Split transcript into chunks
4. Generate embeddings for each chunk using nomic-embed
5. Store embeddings in a Pandas DataFrame
6. For a user query:
   * Generate query embedding
   * Compute cosine similarity with stored embeddings
   * Select top-k most relevant chunks
   * Pass retrieved context to Mistral
   * Generate final answer



# How to Run

Execute "proccessingQuery.py" locally on your system.
After execution, the user can input a question in the CLI.
The system retrieves relevant transcript chunks and generates a context-aware response.



# Chunking Strategy and Observations

Initially, default Whisper transcript segments were used as chunks. However, Whisper often produces very small segments (sometimes single words), which negatively affected retrieval quality due to limited semantic context.
To improve performance, consecutive transcript segments were grouped (5 segments per chunk). This improved contextual coherence and resulted in marginal improvement in retrieval relevance.
This demonstrates the trade-off between chunk granularity and contextual completeness in retrieval systems.



# Retrieval Logic

Cosine similarity (from sklearn.metrics.pairwise) is used to measure semantic similarity between the query embedding and transcript embeddings.
Cosine similarity ensures magnitude-independent comparison in high-dimensional embedding space.
Embeddings are stored in-memory using a Pandas DataFrame(using joblib) rather than a dedicated vector database.
* NOTE : The similarity function from sklearn metrics automatically handles normalization, therefore the vectors are not normalized externally.
  


# Limitations

* Chunk metadata extraction depends on structured filenames
* Embeddings are stored in-memory and are not optimized for large-scale datasets
* Whisper transcription is computationally intensive
* No formal evaluation metric is used to measure retrieval accuracy



# Future Improvements

* Replace in-memory similarity search with FAISS
* Add GPU acceleration for Whisper
* Build a simple web-based interface
* Introduce retrieval evaluation metrics



