"""Module for RAG system operations."""
from typing import List
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, util
import torch

class RAGSystem:
    """Handles Retrieval-Augmented Generation operations."""
    def __init__(self, vectorstore: FAISS):
        """Initialize RAG system with vectorstore."""
        self.vectorstore = vectorstore
        # pylint: disable=not-callable
        self.llm = Ollama(model="llama3.1")
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
        self.reranker = SentenceTransformer('all-MiniLM-L6-v2')
        self.rag_template = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
            {context}
            Question: {question}
            If you do not know the answer, please say you do not know it, 
            DO NOT give answers from outside the context"""
        )
        self.rag_chain = (
            {"context": self._retrieve_and_rerank, "question": RunnablePassthrough()}
            | self.rag_template
            | self.llm
            | StrOutputParser()
        )

    def semantic_rerank(
        self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents based on semantic similarity."""
        query_embedding = self.reranker.encode(query, convert_to_tensor=True)
        doc_embeddings = self.reranker.encode(
            [doc.page_content for doc in documents],
            convert_to_tensor=True
        )
        cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(documents)))
        return [documents[i] for i in top_results.indices]

    def _retrieve_and_rerank(self, query: str) -> str:
        """Retrieve and rerank documents for a query."""
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant documents found"
        unique_docs = []
        seen = set()
        for doc in docs:
            doc_id = repr(doc)
            if doc_id not in seen:
                unique_docs.append(doc)
                seen.add(doc_id)

        reranked_docs = self.semantic_rerank(query, unique_docs)
        return "\n\n".join(doc.page_content for doc in reranked_docs)

    def answer_question(self, question: str) -> str:
        """Generate answer for a given question."""
        return self.rag_chain.invoke(question)

    def retrieve_documents(self, query: str) -> str:
        """Public method to retrieve and rerank documents."""
        return self._retrieve_and_rerank(query)
