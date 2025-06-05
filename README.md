# 🔍 Retrieval-Augmented Generation (RAG) Pipeline for DocNLI

This repository implements a complete Retrieval-Augmented Generation (RAG) pipeline for claim verification using the [DocNLI dataset](https://huggingface.co/datasets/doc_nli). It integrates semantic search and natural language inference (NLI) to determine if a hypothesis is **Supported**, **Refuted**, or **Not Enough Info**.

---

## 📁 Project Structure

- **Dataset**: Subset of DocNLI with `premise`, `hypothesis`, and `label`.
- **Embeddings**: `all-MiniLM-L6-v2` from `sentence-transformers`.
- **Vector Stores**: FAISS (local) and Pinecone (cloud).
- **NLI Model**: `roberta-large-mnli` via HuggingFace Transformers.
- **Optional UI**: Streamlit app with pyngrok (for Colab deployment).

---

## 🚀 Pipeline Overview

1. **Load Data**: Read DocNLI JSON into a Pandas DataFrame.
2. **Embed Premises**: Generate vector embeddings using SentenceTransformers.
3. **Store Embeddings**: Save to FAISS locally or Pinecone for cloud-based search.
4. **Query and Retrieve**: Use vector similarity to fetch the most relevant premise.
5. **Infer Relationship**: Use an NLI model to label the claim as:
   - ✅ **Supported**
   - ❌ **Refuted**
   - ❓ **Not Enough Info**

---

## 🛠️ Installation (Google Colab)

```bash
!pip install sentence-transformers faiss-cpu transformers datasets accelerate pinecone
💡 Example Usage
1. Generate Embedding
python
Copy
Edit
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(["A sample text."])
2. Inference with NLI
python
Copy
Edit
from transformers import pipeline

nli = pipeline("text-classification", model="roberta-large-mnli")
nli("Premise </s></s> Hypothesis")
🎯 Why This Matters
✅ Implements Semantic Search + LLM

✅ Demonstrates Automated Fact-Checking

✅ Great showcase of GenAI + RAG + Vector DB Integration

📚 Author
Shubham Yedekar
📧 yedekarshubham7188@gmail.com
🔗 LinkedIn
📦 More Projects

📜 License
MIT License

yaml
Copy
Edit

---

Let me know when you're ready and I’ll help you upload this to your GitHub repo.

2/2
```







