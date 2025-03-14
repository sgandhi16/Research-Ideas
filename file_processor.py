"""Module for processing different file types into documents."""
from typing import List
from pathlib import Path
import tempfile
import pandas as pd
from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.doc import partition_doc
from unstructured.partition.text import partition_text
from unstructured.partition.docx import partition_docx
from unstructured.cleaners.core import group_broken_paragraphs

# pylint: disable=too-few-public-methods
class FileProcessor:
    """Processes uploaded files into Document objects."""
    @staticmethod
    def process_file(uploaded_file) -> List[Document]:
        """Process a single uploaded file based on its extension."""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'pdf':
            return FileProcessor._process_pdf(uploaded_file)
        if file_extension == 'txt':
            return FileProcessor._process_text(uploaded_file)
        if file_extension in ['docx', 'doc']:
            return FileProcessor._process_doc(uploaded_file, file_extension)
        if file_extension == 'csv':
            return FileProcessor._process_csv(uploaded_file)
        raise ValueError(f"Unsupported file type: {file_extension}")

    @staticmethod
    def _process_pdf(uploaded_file) -> List[Document]:
        """Process PDF files."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            elements = partition_pdf(
                filename=temp_file.name,
                strategy="ocr_only",
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=8000,
                combine_text_under_n_chars=200
            )
            Path(temp_file.name).unlink()
        return [Document(page_content=str(element), metadata={"source": uploaded_file.name})
                for element in elements if hasattr(element, 'text')]

    @staticmethod
    def _process_text(uploaded_file) -> List[Document]:
        """Process text files."""
        text_content = uploaded_file.getvalue().decode("utf-8")
        elements = partition_text(
            text=text_content,
            paragraph_grouper=group_broken_paragraphs,
            max_partition=2500
        )
        return [Document(page_content=str(element), metadata={"source": uploaded_file.name})
                for element in elements if hasattr(element, 'text')]

    @staticmethod
    def _process_doc(uploaded_file, file_extension: str) -> List[Document]:
        """Process DOC/DOCX files."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            elements = (partition_doc if file_extension == 'doc' else partition_docx)(
                filename=temp_file.name,
                chunking_strategy="by_title",
                max_characters=8000,
                strategy="auto",
                infer_table_structure=True,
                combine_text_under_n_chars=800,
                chunk_overlap=500
            )
            Path(temp_file.name).unlink()
        return [Document(page_content=str(element), metadata={"source": uploaded_file.name})
                for element in elements if hasattr(element, 'text')]

    @staticmethod
    def _process_csv(uploaded_file) -> List[Document]:
        """Process CSV files."""
        df = pd.read_csv(uploaded_file)
        return [Document(
            page_content="\n".join(f"{col}: {row[col]}" for col in df.columns),
            metadata={"source": uploaded_file.name, "row": idx}
        ) for idx, row in df.iterrows()]
        