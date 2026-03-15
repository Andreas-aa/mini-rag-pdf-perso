import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data"
INDEX_PATH = "faiss_index"


def load_pdfs():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)


def main():
    print("Loading PDFs...")
    docs = load_pdfs()
    if not docs:
        print(f"Aucun PDF trouvé dans '{DATA_PATH}'.")
        return

    print("Splitting documents...")
    chunks = split_documents(docs)

    print("Creating vector store...")
    create_vector_store(chunks)

    print("Done!")


if __name__ == "__main__":
    main()