import sys
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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


def show_pdf_page(source, page_number):
    doc = fitz.open(source)

    page = doc.load_page(page_number)

    pix = page.get_pixmap()

    image_path = f"preview_page_{page_number}.png"

    pix.save(image_path)

    print(f"Page preview saved: {image_path}")


def ask_question(query):

    vectorstore = load_vector_store()

    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    print("\nTop relevant chunks:\n")

    for i, (doc, score) in enumerate(docs_with_scores, start=1):

        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        print(f"Result {i} | score={score:.4f}")
        print(f"Source: {source} page {page}")

        print("\nChunk preview:")
        print(doc.page_content[:400])
        print("\n--------------------\n")

        show_pdf_page(source, page)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/query.py \"Your question\"")
        sys.exit(1)

    query = sys.argv[1]

    ask_question(query)