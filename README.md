# Fusion_RAG
🧠 Why Fusion RAG?
Simple RAG retrieves documents using the original query only, which might miss relevant info due to:

Typos

Ambiguous wording

Missing synonyms

Fusion RAG solves this by:

Generating alternative phrasings of the question.

Running retrieval for each phrasing.

Combining and ranking the results.

📦 Use Cases
Use Case	Why Fusion RAG Helps
🧑‍🏫 Academic Q&A	Captures more complete context for vague questions
🏛️ Legal Search	Retrieves semantically similar documents
📚 Research Assistant	Pulls broader range of citations
🧾 Policy Advisor	Gets nuanced matching from legal/policy docs

🛠️ Tech Stack
Groq LLM via ChatGroq

LangChain for chaining and multi-query fusion

HuggingFace Embeddings

FAISS as vector store

Streamlit for UI
