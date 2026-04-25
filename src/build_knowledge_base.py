"""
One-shot script to build the RAG vector store from the disease KB JSON.

Run with:
    python -m src.build_knowledge_base

This embeds all 38 disease entries and persists the Chroma DB under
models/chroma_db/. You only need to run this once (or whenever you update
the knowledge base).
"""

from pathlib import Path
import sys
from src.rag import load_knowledge_base, build_vectorstore, DEFAULT_PERSIST_DIR


def main():
    kb_path = Path("data/disease_knowledge_base.json")
    if not kb_path.exists():
        print(f"Error: {kb_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading knowledge base from {kb_path}...")
    docs = load_knowledge_base(str(kb_path))
    print(f"Loaded {len(docs)} disease documents.")

    print(f"Building vector store at {DEFAULT_PERSIST_DIR}...")
    vs = build_vectorstore(docs, persist_directory=DEFAULT_PERSIST_DIR)
    print(f"Done. Vector store contains {vs._collection.count()} embeddings.")


if __name__ == "__main__":
    main()
