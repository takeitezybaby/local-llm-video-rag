import json
import os
import requests
import chromadb
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# 1. Connect to ChromaDB
client = chromadb.PersistentClient(path="Data/chromaDB")
collection = client.get_collection(name="video_transcripts")

def createEmbeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "nomic-embed-text",
        "input": text_list
    })
    return r.json()["embeddings"]

def generate_llm_response(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral:7b",
        "prompt": prompt,
        "stream": False
    })
    return r.json()["response"]

def run_rag_pipeline(question, top_k=5):
    query_emb = createEmbeddings([question])
    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k
    )
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    context_list = []
    for meta, doc in zip(metas, docs):
        context_list.append({
            "Video number": meta.get("video_number", 0),
            "Title": meta.get("Title", ""),
            "start_time": meta.get("start_time", 0.0),
            "end_time": meta.get("end_time", 0.0),
            "text": doc
        })
    
    json_context = json.dumps(context_list, indent=2)
    
    prompt = f"""You are a friendly, knowledgeable Python tutor helping a student navigate their course videos. Talk directly to the student in a warm, encouraging, and natural conversational tone (use "you" and "your", never refer to them as "the student").

=== VIDEO TRANSCRIPT CONTEXT ===
{json_context}

=== USER QUESTION ===
{question}

=== GUIDELINES ===
1. Directly answer their question in 1-2 simple, friendly sentences.
2. Naturally guide them to the exact video and timestamp range (MM:SS format).
3. Base your help ONLY on the provided video transcripts.
4. Do NOT ask follow-up questions at the end."""

    answer = generate_llm_response(prompt)
    return answer, docs

def main():
    print("=" * 60, flush=True)
    print("       RAGAS BENCHMARK EVALUATOR FOR VIDEO RAG", flush=True)
    print("=" * 60, flush=True)
    
    dataset_path = "Data/eval_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.", flush=True)
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    print(f"Loaded {len(eval_data)} benchmark test cases.\n", flush=True)
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for idx, item in enumerate(eval_data, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"[{idx}/{len(eval_data)}] Running RAG for: '{q}'", flush=True)
        
        ans, retrieved_contexts = run_rag_pipeline(q, top_k=5)
        
        questions.append(q)
        answers.append(ans)
        contexts.append(retrieved_contexts)
        ground_truths.append(gt)
    
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    print("\n" + "=" * 60, flush=True)
    print("Initializing Local Ollama Evaluators (Mistral 7B + Nomic-Embed)...", flush=True)
    print("=" * 60, flush=True)
    
    eval_llm = LangchainLLMWrapper(ChatOllama(model="mistral:7b", temperature=0.0))
    eval_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    
    # Configure RunConfig for local sequential Ollama execution to prevent timeouts
    run_config = RunConfig(
        max_workers=1,
        timeout=300,
        max_retries=5,
        max_wait=60
    )
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
    
    print("Running Ragas metric calculations sequentially to ensure stability...", flush=True)
    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config
    )
    
    print("\n" + "=" * 60, flush=True)
    print("                 BENCHMARK RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(results, flush=True)
    
    df_results = results.to_pandas()
    os.makedirs("Data", exist_ok=True)
    os.makedirs("Sample_outputs", exist_ok=True)
    
    csv_path = "Data/evaluation_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nDetailed CSV exported to: {csv_path}", flush=True)
    
    # Save a clean markdown summary report
    report_path = "Sample_outputs/benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RAGAS Benchmark Report - Video RAG System\n\n")
        f.write("## Overall Metrics Summary\n\n")
        f.write("| Metric | Score |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Faithfulness** | `{df_results['faithfulness'].mean():.4f}` |\n")
        f.write(f"| **Answer Relevancy** | `{df_results['answer_relevancy'].mean():.4f}` |\n")
        f.write(f"| **Context Precision** | `{df_results['context_precision'].mean():.4f}` |\n")
        f.write(f"| **Context Recall** | `{df_results['context_recall'].mean():.4f}` |\n")
        f.write("\n---\n\n## Per-Question Breakdown\n\n")
        f.write(df_results[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]].to_markdown(index=False))
    
    print(f"Markdown report generated: {report_path}", flush=True)

if __name__ == "__main__":
    main()
