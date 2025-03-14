"""Module for managing vectorstore operations."""
from typing import List, Set, Optional
from pathlib import Path
import json
from langchain_community.vectorstores import FAISS
from langchain_community import embeddings
from langchain.schema import Document
import streamlit as st
from file_processor import FileProcessor
class VectorStoreManager:
    """Manages vectorstore creation and persistence."""
    PERSISTENCE_DIR = Path("./demostore")
    VECTORSTORE_PATH = PERSISTENCE_DIR / "faiss_index"
    PROCESSED_FILES_PATH = PERSISTENCE_DIR / "processed_files.json"

    def __init__(self):
        """Initialize embeddings model."""
        self.embeddings_model = embeddings.OllamaEmbeddings(model="nomic-embed-text")

    def load_processed_files(self) -> Set[str]:
        """Load previously processed file names."""
        if self.PROCESSED_FILES_PATH.exists():
            with open(self.PROCESSED_FILES_PATH, 'r', encoding='utf-8') as file:
                return set(json.load(file))
        return set()

    def save_processed_files(self, processed_files: Set[str]) -> None:
        """Save processed file names."""
        with open(self.PROCESSED_FILES_PATH, 'w', encoding='utf-8') as file:
            json.dump(list(processed_files), file)

    def load_vectorstore(self) -> Optional[FAISS]:
        """Load existing vectorstore or return None."""
        if self.VECTORSTORE_PATH.exists():
            try:
                return FAISS.load_local(
                    self.VECTORSTORE_PATH,
                    self.embeddings_model,
                    allow_dangerous_deserialization=True
                )
            except ValueError as error:
                st.error(f"Error loading vectorstore: {str(error)}")
                st.warning("Creating new vectorstore due to loading failure.")
                return None
        return None

    def save_vectorstore(self, vectorstore: FAISS) -> None:
        """Save vectorstore to disk."""
        self.PERSISTENCE_DIR.mkdir(exist_ok=True)
        vectorstore.save_local(self.VECTORSTORE_PATH)

    def update_vectorstore(self, uploaded_files, processed_files: Set[str],
                         vectorstore: Optional[FAISS] = None) -> FAISS:
        """Update vectorstore with new documents."""
        new_documents: List[Document] = []
        for file in uploaded_files:
            if file.name not in processed_files:
                processed_docs = FileProcessor.process_file(file)
                new_documents.extend(processed_docs)
                processed_files.add(file.name)

        if vectorstore is None:
            vectorstore = FAISS.from_documents(
                new_documents or [Document(page_content="Placeholder document")],
                self.embeddings_model
            )
        elif new_documents:
            vectorstore.add_documents(new_documents)

        if new_documents:
            self.save_vectorstore(vectorstore)
            self.save_processed_files(processed_files)

        return vectorstore
