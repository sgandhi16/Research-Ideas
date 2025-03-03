#Fact checking done using another LLM
import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community import embeddings
from langchain.schema import Document
import os
import tempfile
import pandas as pd
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.doc import partition_doc
from unstructured.partition.text import partition_text
from unstructured.cleaners.core import group_broken_paragraphs
from unstructured.partition.docx import partition_docx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from io import BytesIO
from typing import List, Set
from sentence_transformers import SentenceTransformer, util
import torch
import json

from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'show_faq' not in st.session_state:
    st.session_state.show_faq = False
if 'show_infra_faq' not in st.session_state:
    st.session_state.show_infra_faq = False    
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'temp_input' not in st.session_state:
    st.session_state.temp_input = ""
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = None
if 'top_chunks' not in st.session_state:
    st.session_state.top_chunks = []

def process_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        elements = partition_pdf(filename=temp_file_path, strategy="ocr_only", infer_table_structure=True,chunking_strategy="by_title",max_characters= 8000, combine_text_under_n_chars=200)
        os.unlink(temp_file_path)
        texts = [str(element) for element in elements if hasattr(element, 'text')]
        return [Document(page_content=text, metadata={"source": uploaded_file.name}) for text in texts]
    
    elif file_extension == 'txt':
        text_content = uploaded_file.getvalue().decode("utf-8")
        elements = partition_text(text=text_content,paragraph_grouper=group_broken_paragraphs, max_partition=2500)
        texts = [str(element) for element in elements if hasattr(element, 'text')]
        return [Document(page_content=text, metadata={"source": uploaded_file.name}) for text in texts]
   
    elif file_extension in ['docx', 'doc']:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        if file_extension == 'doc':
            elements = partition_doc(filename=temp_file_path, chunking_strategy="by_title", max_characters=8000, strategy="auto", infer_table_structure=True, combine_text_under_n_chars=800, chunk_overlap=500)
        else:  # docx
            elements = partition_docx(filename=temp_file_path, chunking_strategy="by_title", max_characters=8000, strategy="auto", infer_table_structure=True, combine_text_under_n_chars=800, chunk_overlap=500)
        
        os.unlink(temp_file_path)
        texts = [str(element) for element in elements if hasattr(element, 'text')]
        return [Document(page_content=text, metadata={"source": uploaded_file.name}) for text in texts]
    
    elif file_extension == 'csv':
        df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
        documents = []
        for _, row in df.iterrows():
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            doc = Document(
                page_content=content,
                metadata={"source": uploaded_file.name, "row": _}
            )
            documents.append(doc)
        return documents
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def read_faq_file(file_path):
    questions = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue
            elif line:
                questions.append((line))
    return questions

def read_infra_faq_file(file_path):
    questions = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue
            elif line:
                questions.append((line))
    return questions
    
def set_question(question):
    st.session_state.temp_input = question
    
def semantic_rerank(query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode([doc.page_content for doc in documents], convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(documents)))
    return [documents[i] for i in top_results.indices]

# Function to calculate confidence score using LLM 2
def calculate_confidence_score(llm_answer, context_chunks):
    # Initializing second LLM
    fact_check_llm = Ollama(model="llama3.1") #first install this model by executing the command "ollama run llama3.1" on your Terminal
    
    # Combine the chunks into a single context string
    context = "\n\n".join([chunk.page_content for chunk in context_chunks])
    
    # Create prompt for fact-checking
    fact_check_template = """
    You are a fact-checking assistant. Your task is to evaluate the accuracy of an answer based on the provided context.

    Context:
    {context}

    LLM Answer:
    {answer}

    Instructions:
    1. Analyze if the answer is fully supported by the context
    2. Check for any factual inconsistencies, hallucinations, or incorrect information
    3. Provide a confidence score (0-100) where:
       - 90-100: Answer is completely accurate and fully supported by the context
       - 70-89: Answer is mostly accurate with minor issues or small inferences
       - 50-69: Answer has some factual basis but contains unsupported claims
       - 30-49: Answer has significant inconsistencies with the context
       - 0-29: Answer contains major hallucinations or is contradicted by the context
    
    Format your response as follows:
    Confidence Score: [Number between 0-100]
    Explanation: [Brief explanation of your assessment]
    """
    
    fact_check_prompt = ChatPromptTemplate.from_template(fact_check_template)
    
    # Create chain for fact checking
    fact_check_chain = (
        fact_check_prompt 
        | fact_check_llm 
        | StrOutputParser()
    )
    
    # Run the fact checking
    fact_check_result = fact_check_chain.invoke({
        "context": context,
        "answer": llm_answer
    })
    
    # Extract the confidence score (assuming format "Confidence Score: XX")
    try:
        score_line = [line for line in fact_check_result.split('\n') if 'Confidence Score:' in line][0]
        score = int(score_line.split(':')[1].strip().split()[0])
        explanation = fact_check_result.split('Explanation:')[1].strip() if 'Explanation:' in fact_check_result else ""
        return score, explanation
    except (IndexError, ValueError):
        # Fallback if parsing fails
        return 50, "Unable to parse confidence score properly. Review manually."

PERSISTENCE_DIR = Path("./demostore")#Define as per your need
VECTORSTORE_PATH = PERSISTENCE_DIR / "faiss_index"
PROCESSED_FILES_PATH = PERSISTENCE_DIR / "processed_files.json"

def save_processed_files(processed_files: Set[str]):
    with open(PROCESSED_FILES_PATH, "w") as f:
        json.dump(list(processed_files), f)

def load_processed_files() -> Set[str]:
    if PROCESSED_FILES_PATH.exists():
        with open(PROCESSED_FILES_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_vectorstore(vectorstore: FAISS):
    PERSISTENCE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

def load_vectorstore() -> FAISS:
    if VECTORSTORE_PATH.exists():
        embeddings_model = embeddings.OllamaEmbeddings(model="nomic-embed-text") #Pull this model by running "ollama run nomic-embed-text"
        try:
            return FAISS.load_local(VECTORSTORE_PATH, embeddings_model, allow_dangerous_deserialization=True)
        except ValueError as e:
            st.error(f"Error loading vectorstore: {str(e)}")
            st.warning("The existing vectorstore could not be loaded safely. A new one will be created.")
            return None
    return None

def process_and_update_vectorstore(uploaded_files, processed_files: Set[str], vectorstore: FAISS = None) -> FAISS:
    embeddings_model = embeddings.OllamaEmbeddings(model="nomic-embed-text")
    new_documents = []

    for file in uploaded_files:
        if file.name not in processed_files:
            processed_docs = process_file(file)
            new_documents.extend(processed_docs)
            processed_files.add(file.name)

    if vectorstore is None:
        if new_documents:
            vectorstore = FAISS.from_documents(documents=new_documents, embedding=embeddings_model)
        else:
            vectorstore = FAISS.from_documents(documents=[Document(page_content="Placeholder document")], embedding=embeddings_model)
    elif new_documents:
        vectorstore.add_documents(new_documents)

    if new_documents:
        save_vectorstore(vectorstore)
        save_processed_files(processed_files)

    return vectorstore
    
def main():
    st.title("Multi-Format RAG System with Fact-Checking")

    processed_files = load_processed_files()
    vectorstore = load_vectorstore()

    if vectorstore is None:
        st.warning("A new vectorstore will be created for newly uploaded files.")

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf", "txt", "docx", "doc", "csv"])

    if uploaded_files:
        vectorstore = process_and_update_vectorstore(uploaded_files, processed_files, vectorstore)

        llm = Ollama(model="llama3.1")
        embeddings_model = embeddings.OllamaEmbeddings(model="nomic-embed-text")

        new_documents = []
        for file in uploaded_files:
            if file.name not in processed_files:
                processed_docs = process_file(file)
                new_documents.extend(processed_docs)
                processed_files.add(file.name)

        if vectorstore is None:
            if new_documents:
                vectorstore = FAISS.from_documents(documents=new_documents, embedding=embeddings_model)
            else:
                vectorstore = FAISS.from_documents(documents=[Document(page_content="Placeholder document")], embedding=embeddings_model)
        elif new_documents:
            vectorstore.add_documents(new_documents)

        # Save updated vectorstore and processed files
        save_vectorstore(vectorstore)
        save_processed_files(processed_files)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

        after_rag_template = """Answer the question based only on the following context:
        {context} 
        Question: {question}
        If you do not know the answer, please say you do not know it, DO NOT give answers from outside the context
        """
        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def retrieve_and_rerank(query):
            
            docs = retriever.get_relevant_documents(query)
            if not docs:
                st.warning("No relevant documents found for the query!")
            
            seen = set()
            combined_results = []
            for doc in docs:
                doc_identifier = repr(doc)
                if doc_identifier not in seen:
                    combined_results.append(doc)
                    seen.add(doc_identifier)

            # Perform semantic re-ranking
            reranked_docs = semantic_rerank(query, combined_results, top_k=5)
            
            # Debugging print
            st.write("Debug: Retrieved and Reranked Documents Count:", len(reranked_docs))

            # Store the top chunks for fact-checking
            st.session_state.top_chunks = reranked_docs
            
            return format_docs(reranked_docs)


        rag_chain = (
            {"context": retrieve_and_rerank, "question": RunnablePassthrough()}
            | after_rag_prompt
            | llm
            | StrOutputParser()
        )

        st.subheader("Chat")
        col1, col2 = st.columns([6, 1])

        with col1:
            user_input = st.text_input("Your question", label_visibility="collapsed", placeholder="Enter your question", key="user_input", value=st.session_state.temp_input)

        with col2:
            submit_button = st.button("Submit")   
            
        col3, col4 = st.columns([1, 3])

        with col3:
            faq_button = st.button("List of FAQs")

           

        if faq_button:
            st.session_state.show_faq = not st.session_state.show_faq
            
        
        if st.session_state.show_faq:
            with st.expander("Frequently Asked Questions", expanded=True):
               questions = read_faq_file("./cricket_faqs.txt")
               for question in questions:
                    if st.button(question, key=f"faq_{question}", on_click=set_question, args=(question,)):
                        pass
                                                      

        if 'llm_response' not in st.session_state:
            st.session_state.llm_response = ""

        if submit_button and user_input:
            with st.spinner("Generating answer..."):
                # Invoke the RAG chain with the user's input
                st.session_state.llm_response = rag_chain.invoke(user_input)
                #st.session_state.user_input = user_input

            # Display the answer
            st.markdown(f"""
                <div style="background-color: #d1f4f3; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong style="color: #007b7f;">User:</strong> {user_input}
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="background-color: #e9f7ef; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong style="color: #34a853;">Assistant:</strong> {st.session_state.llm_response}
                </div>
            """, unsafe_allow_html=True)

            # Calculate confidence score using LLM 2
            with st.spinner("Fact-checking..."):
                score, explanation = calculate_confidence_score(
                    st.session_state.llm_response, 
                    st.session_state.top_chunks
                )
                st.session_state.confidence_score = score
                st.session_state.confidence_explanation = explanation

            # Display confidence score with appropriate color
            if score >= 90:
                confidence_color = "#34a853"  # Green for high confidence
            elif score >= 70:
                confidence_color = "#4285f4"  # Blue for good confidence
            elif score >= 50:
                confidence_color = "#fbbc05"  # Yellow for medium confidence
            else:
                confidence_color = "#ea4335"  # Red for low confidence
                
            st.markdown(f"""
                <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 10px; border-left: 5px solid {confidence_color};">
                    <strong style="color: {confidence_color};">Confidence Score: {score}/100</strong>
                    <p>{explanation}</p>
                </div>
            """, unsafe_allow_html=True)

            # Set the flag to indicate the user has submitted a question
            st.session_state.user_submitted = True

        # Initialize the 'show_logs' flag in session state if it doesn't exist
        if 'show_logs' not in st.session_state:
            st.session_state.show_logs = False  # Initialize to False

        # Display the toggle button only if the user has submitted a question
        if 'user_submitted' in st.session_state and st.session_state.user_submitted:
            if st.button("Show/Hide Logs"):
                st.session_state.show_logs = not st.session_state.show_logs

            # Display the original LLM response along with the logs if 'show_logs' is True
            if st.session_state.show_logs:
                st.write("Original LLM Response:")
                st.markdown(f"""
                    <div style="background-color: #e9f7ef; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong style="color: #34a853;">Assistant:</strong> {st.session_state.llm_response}
                    </div>
                """, unsafe_allow_html=True)

                st.write("Top 5 semantically re-ranked matches from vector store:")

                results= retriever.get_relevant_documents(st.session_state.user_input)
                reranked_results = semantic_rerank(st.session_state.user_input, results, top_k=5)

                # Use the stored top chunks
                for i, doc in enumerate(reranked_results, 1):
                    st.write(f"{i}. {doc.page_content[:500]}...")


if __name__ == "__main__":
    main()