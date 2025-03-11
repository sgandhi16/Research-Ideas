"""Main application module."""
from typing import List
import streamlit as st
from vectorstore_manager import VectorStoreManager
from fact_checker import FactChecker
from rag_system import RAGSystem

def initialize_session_state() -> None:
    """Initialize session state variables."""
    defaults = {
        'chat_history': [],
        'vectorstore': None,
        'processed_files': set(),
        'show_faq': False,
        'evaluation_results': [],
        'temp_input': "",
        'confidence_score': None,
        'top_chunks': [],
        'user_submitted': False,
        'show_logs': False,
        'llm_response': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def read_faq_file(file_path: str) -> List[str]:
    """Read FAQ questions from a file."""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                questions.append(line)
    return questions

def display_confidence_score(score: int, explanation: str) -> None:
    """Display confidence score with appropriate styling."""
    color_map = {
        range(90, 101): "#34a853",  # Green
        range(70, 90): "#4285f4",   # Blue
        range(50, 70): "#fbbc05",   # Yellow
        range(0, 50): "#ea4335"     # Red
    }
    color = next(c for r, c in color_map.items() if score in r)
    st.markdown(f"""
        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 10px; border-left: 5px solid {color};">
            <strong style="color: {color};">Confidence Score: {score}/100</strong>
            <p>{explanation}</p>
        </div>
    """, unsafe_allow_html=True)

def main() -> None:
    """Main application function."""
    st.title("Multi-Format RAG System with Fact-Checking")
    initialize_session_state()

    vector_manager = VectorStoreManager()
    fact_checker = FactChecker()
    processed_files = vector_manager.load_processed_files()
    vectorstore = vector_manager.load_vectorstore()
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "doc", "csv"]
    )

    if uploaded_files:
        vectorstore = vector_manager.update_vectorstore(
            uploaded_files, processed_files, vectorstore)
        rag_system = RAGSystem(vectorstore)
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Your question",
                label_visibility="collapsed",
                placeholder="Enter your question",
                value=st.session_state.temp_input
            )
        with col2:
            submit_button = st.button("Submit")

        col3, _ = st.columns([1, 3])
        with col3:
            faq_button = st.button("List of FAQs")

        if faq_button:
            st.session_state.show_faq = not st.session_state.show_faq

        if st.session_state.show_faq:
            with st.expander("Frequently Asked Questions", expanded=True):
                for question in read_faq_file("./cricket_faqs.txt"):
                    st.button(question, on_click=lambda q=question: setattr(
                        st.session_state, 'temp_input', q))

        if submit_button and user_input:
            with st.spinner("Generating answer..."):
                response = rag_system.answer_question(user_input)
                st.session_state.llm_response = response
            st.markdown(f"""
                <div style="background-color: #d1f4f3; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong style="color: #007b7f;">User:</strong> {user_input}
                </div>
                <div style="background-color: #e9f7ef; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong style="color: #34a853;">Assistant:</strong> {response}
                </div>
            """, unsafe_allow_html=True)

            with st.spinner("Fact-checking..."):
                score, explanation = fact_checker.calculate_confidence_score(
                    response,
                    rag_system.semantic_rerank(
                        user_input, rag_system.retriever.get_relevant_documents(user_input))
                )
                st.session_state.confidence_score = score
                st.session_state.confidence_explanation = explanation

            display_confidence_score(score, explanation)
            st.session_state.user_submitted = True

        if st.session_state.user_submitted:
            if st.button("Show/Hide Logs"):
                st.session_state.show_logs = not st.session_state.show_logs

            if st.session_state.show_logs:
                st.write("Original LLM Response:")
                st.markdown(f"""
                    <div style="background-color: #e9f7ef; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong style="color: #34a853;">Assistant:</strong> {st.session_state.llm_response}
                    </div>
                """, unsafe_allow_html=True)
                st.write("Top 5 semantically re-ranked matches:")
                results = rag_system.retriever.get_relevant_documents(user_input)
                reranked = rag_system.semantic_rerank(user_input, results)
                for i, doc in enumerate(reranked, 1):
                    st.write(f"{i}. {doc.page_content[:500]}...")

if __name__ == "__main__":
    main()
