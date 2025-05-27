import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

# LLM via Groq
llm = ChatGroq(
    groq_api_key="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",
    model_name="llama3-8b-8192"
)

# Set up multi-query retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=llm
)

# Prompt template for answer generation
fusion_prompt = PromptTemplate.from_template("""
Use the following context from multiple sources to answer the question clearly and concisely.

Context:
{context}

Question: {question}
Answer:
""")

fusion_chain = LLMChain(llm=llm, prompt=fusion_prompt)

# Streamlit UI
st.title("🧠 Fusion RAG with Groq")
query = st.text_input("Ask your question:")

if query:
    # Step 1: Get multiple document sets
    docs = multi_query_retriever.get_relevant_documents(query)

    # Step 2: Fuse context
    fused_context = "\n\n".join([doc.page_content for doc in docs])

    # Step 3: Generate answer
    answer = fusion_chain.run({"context": fused_context, "question": query})

    # Output
    st.subheader("💡 Answer")
    st.write(answer)

    with st.expander("📄 Retrieved Documents"):
        for doc in docs:
            st.markdown(doc.page_content)
