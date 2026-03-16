import sys
import ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = "faiss_index"


def load_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def ask_question(question):

    vectorstore = load_vector_store()

    docs_with_scores = vectorstore.similarity_search_with_score(question, k=5)

    docs = [doc for doc, score in docs_with_scores]

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a question answering assistant.

Answer ONLY using the context.

If the answer is not in the context say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAnswer:\n")
    print(response["message"]["content"])

    print("\nRetrieved sources:\n")

    for i, (doc, score) in enumerate(docs_with_scores, start=1):

        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        print(f"{i}. {source} page {page} | score={score:.4f}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python rag_query.py \"Your question\"")
        sys.exit(1)

    question = sys.argv[1]

    ask_question(question)