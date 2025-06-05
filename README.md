Retrieval-Augmented Generation (RAG) Pipeline for DocNLI
This project implements a RAG pipeline using the DocNLI dataset. It combines document retrieval and natural language inference (NLI) to verify claims based on a corpus of premises.

📌 Project Structure
Dataset: Subset of the DocNLI dataset (premise, hypothesis, label).

Embedding Model: all-MiniLM-L6-v2 via SentenceTransformers.

Vector Store Options: FAISS (local), Pinecone (cloud).

NLI Model: roberta-large-mnli via HuggingFace Transformers.

Interface (Optional): Streamlit (with pyngrok for Google Colab).

🚀 Pipeline Overview
Load & Preprocess Data: JSON → Pandas DataFrame.

Embed Premises: SentenceTransformer → Vector Embeddings.

Store Vectors: Using FAISS or Pinecone.

Retrieve Relevant Premise: Based on semantic similarity.

Infer NLI Result: Classify relation between premise and hypothesis as:

Supported

Refuted

Not Enough Info

🛠️ Installation (Colab)
bash
Copy
Edit
!pip install sentence-transformers faiss-cpu transformers datasets accelerate pinecone
💡 Example Usage
python
Copy
Edit
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(["A sample text."])
python
Copy
Edit
from transformers import pipeline
nli = pipeline("text-classification", model="roberta-large-mnli")
nli("Premise </s></s> Hypothesis")
🧠 Why This Matters
✅ Semantic document retrieval

✅ Automated fact-checking with LLM

✅ End-to-end RAG on real-world data

