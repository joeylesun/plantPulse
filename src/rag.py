"""
RAG pipeline for plant disease treatment advice.

Rubric item addressed: RAG via LangChain (treatment advice retrieval).

Architecture:
    disease_knowledge_base.json  (one document per disease)
        -> sentence-transformer embeddings  (all-MiniLM-L6-v2, runs locally, free)
        -> ChromaDB vector store  (persistent on disk, no cloud dependency)
        -> retrieve top-k chunks for the predicted disease
        -> OpenAI GPT-4o-mini (or any chat model) synthesizes the advice

We use HuggingFace embeddings instead of OpenAI embeddings so the vector
store can be built without internet or API costs; only the final generation
call needs an OpenAI key.

AI attribution: Structure drafted with Claude (Anthropic), adapted by author.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import os

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "models/chroma_db"


def load_knowledge_base(json_path: str) -> List[Document]:
    """Load the disease KB JSON into LangChain Documents.

    Each entry in the JSON maps a disease class name (matching the folder names
    in PlantVillage, e.g. 'Tomato___Early_blight') to a dict with keys:
        disease, plant, symptoms, causes, treatment, prevention
    We turn each into a single text Document with structured metadata.
    """
    with open(json_path, "r") as f:
        kb = json.load(f)

    docs = []
    for class_name, info in kb.items():
        # Flat text chunk - tends to work better than splitting fields across
        # multiple chunks for disease info, which is short.
        content = (
            f"Plant: {info['plant']}\n"
            f"Disease: {info['disease']}\n"
            f"Symptoms: {info['symptoms']}\n"
            f"Causes: {info['causes']}\n"
            f"Treatment: {info['treatment']}\n"
            f"Prevention: {info['prevention']}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "class_name": class_name,
                    "plant": info["plant"],
                    "disease": info["disease"],
                },
            )
        )
    return docs


def build_vectorstore(
    docs: List[Document],
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """Embed docs with a local sentence-transformer and persist to disk."""
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vs


def load_vectorstore(persist_directory: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """Reload an existing persisted vector store."""
    if not Path(persist_directory).exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_directory}. "
            f"Build it first with: python -m src.build_knowledge_base"
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


# Prompt template - tells the LLM to stay grounded in retrieved context
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are PlantDoc, a helpful assistant specialized in plant disease "
     "diagnosis and treatment. Use ONLY the retrieved context below to answer. "
     "If the context does not contain the answer, say you don't have that "
     "information rather than guessing. Keep responses concise and actionable - "
     "structure them as: Quick Diagnosis, Immediate Actions, Long-term Prevention.\n\n"
     "Retrieved context:\n{context}"),
    ("human", "{question}"),
])


class PlantDocRAG:
    """RAG system tying together retrieval + generation."""

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        openai_model: str = "gpt-4o-mini",
        k: int = 3,
    ):
        self.vectorstore = load_vectorstore(persist_directory)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self.llm = ChatOpenAI(model=openai_model, temperature=0.2)
        self.prompt = RAG_PROMPT

    def retrieve(self, query: str, class_filter: Optional[str] = None) -> List[Document]:
        """Retrieve top-k docs. If class_filter is set, restrict to that disease."""
        if class_filter:
            # Direct metadata lookup - much more accurate than semantic search
            # when we already know exactly which disease we're asking about.
            return self.vectorstore.similarity_search(
                query, k=self.retriever.search_kwargs["k"],
                filter={"class_name": class_filter},
            )
        return self.retriever.invoke(query)

    def answer(self, question: str, class_filter: Optional[str] = None) -> Dict:
        """Full RAG: retrieve, then generate an answer with citations."""
        docs = self.retrieve(question, class_filter=class_filter)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        chain = self.prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        return {
            "answer": response.content,
            "sources": [d.metadata for d in docs],
            "context": context,
        }


def advice_for_predicted_disease(
    rag: PlantDocRAG,
    predicted_class: str,
) -> Dict:
    """Convenience wrapper: given a predicted class name from the CNN, ask the
    RAG system for a treatment plan scoped to that specific disease."""
    pretty = predicted_class.replace("___", " - ").replace("_", " ")
    question = (
        f"My plant has been diagnosed with {pretty}. What are the symptoms I "
        f"should confirm, immediate treatment steps, and long-term prevention?"
    )
    return rag.answer(question, class_filter=predicted_class)
