"""Module for fact-checking LLM responses."""
from typing import Tuple, List
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# pylint: disable=too-few-public-methods
class FactChecker:
    """Handles fact-checking of LLM responses."""
    def __init__(self):
        """Initialize fact-checking LLM."""
        # pylint: disable=not-callable
        self.fact_check_llm = Ollama(model="llama3.1")
        self.fact_check_template = ChatPromptTemplate.from_template("""
            You are a fact-checking assistant. Your task is to evaluate the accuracy of an answer based on the provided context.

            Context:
            {context}

            LLM Answer:
            {answer}

            Instructions:
            1. Analyze if the answer is fully supported by the context
            2. Check for any factual inconsistencies, hallucinations, or incorrect information
            3. Provide a confidence score (0-100) where:
               - 90-100: Answer is completely accurate and fully supported
               - 70-89: Answer is mostly accurate with minor issues
               - 50-69: Answer has some factual basis but contains unsupported claims
               - 30-49: Answer has significant inconsistencies
               - 0-29: Answer contains major hallucinations or is contradicted
            
            Format your response as follows:
            Confidence Score: [Number between 0-100]
            Explanation: [Brief explanation]
        """)
        self.fact_check_chain = (
            self.fact_check_template
            | self.fact_check_llm
            | StrOutputParser()
        )

    def calculate_confidence_score(
            self,
            llm_answer: str,
            context_chunks: List[Document]
        ) -> Tuple[int, str]:
        """Calculate confidence score for an LLM answer."""
        context = "\n\n".join(chunk.page_content for chunk in context_chunks)
        result = self.fact_check_chain.invoke({"context": context, "answer": llm_answer})
        try:
            score_line = next(line for line in result.split('\n') if 'Confidence Score:' in line)
            score = int(score_line.split(':')[1].strip().split()[0])
            if 'Explanation:' in result:
                explanation = result.split('Explanation:')[1].strip()
            else:
                explanation = ""

            return score, explanation
        except (IndexError, ValueError):
            return 50, "Unable to parse confidence score properly. Review manually."
        