# Simple RAG Project (PDF Question Answering)

## Overview

This project implements a **simple Retrieval-Augmented Generation (RAG) system** using Python.

The goal is to allow a user to **ask questions about a set of PDF documents**. The system retrieves the most relevant parts of the documents and uses a local language model to generate an answer.

The pipeline works as follows:

1. PDF documents are loaded from the `data/` folder.
2. The text is split into smaller chunks.
3. Each chunk is converted into a vector embedding.
4. The vectors are stored in a FAISS vector database.
5. When a user asks a question, the system finds the most relevant chunks.
6. These chunks are sent to a local LLM (via Ollama) to generate the final answer.

---

## Documents Used

The project uses three example PDF documents stored in the `data/` folder.

### Football PDF

This document describes the sport of football.
It explains the basic rules of the game, how teams score goals, and the roles of players such as defenders, midfielders, and forwards. It may also mention competitions, teamwork, and the popularity of football around the world.

---

### Wildlife PDF

This document focuses on wildlife and animals living in nature.
It describes different species, their habitats, behaviors, and the ecosystems where they live. It may include examples such as mammals, birds, and other animals found in forests or natural environments.

---

### Sea PDF

This document describes the marine environment.
It explains aspects of the ocean such as sea animals, underwater ecosystems, coral reefs, and marine biodiversity. It highlights how many different organisms live in the sea and how the ocean supports life on Earth.

---

## Usage

1. Place the PDFs in the `data/` folder.
2. Run the indexing script to build the vector database.
3. Ask questions about the documents using the query script.

Example:

```bash
python rag_query.py "What animals live in the ocean?"
```

The system will retrieve the most relevant parts of the PDFs and generate an answer based on them.
