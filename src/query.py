import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = "faiss_index"


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def ask_question(query):
    vectorstore = load_vector_store()
    docs = vectorstore.similarity_search(query, k=3)

    print("\nTop relevant chunks:\n")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        print(f"Result {i+1}")
        print(f"Source: {source} page {page}")
        print(doc.page_content[:500])
        print("\n------------------\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/query.py \"Your question\"")
        sys.exit(1)
    query = sys.argv[1]
    ask_question(query)