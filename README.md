# Research-Ideas
Multimodal RAG App for querying documents 

## Multi-modal RAG system with Feedback
This project is meant to provide a framework where users can get answers based on questions asked from pre-defined documents in PDF, TXT, DOCX, CSV formats.
Documents uploaded are converted to vectors using embedding models and then reranked based on the query before feeding to LLM to get the final output . 
Brief description of the various topics are given below along with the instructions to set up a running application.

## Retrieval Augmented Generation (RAG)
Retrieval Augmented Generation (RAG), is a technique that combines the power of large language models (LLMs) with the ability to retrieve relevant information from external knowledge sources. This means an LLM doesn't just rely on its internal training data, but can also access and incorporate up-to-date, specific information when generating responses.
For eg. Functional design specs, runbooks and other types of documentation pages can be provided to a RAG system to augment the LLM's knowledge.

(Reference: https://www.oracle.com/in/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/#:~:text=Retrieval%2Daugmented%20generation%20is%20a,already%20contained%20in%20the%20LLM.)

## About Ollama
Ollama is an open-source project that serves as a powerful and user-friendly platform for running LLMs on your local machine.
It simplifies the process of downloading, installing, and interacting with a wide range of LLMs, empowering users to explore their capabilities without the need for extensive technical expertise or reliance on cloud-based platforms. In this project we use Ollama's "llama3.1" model

(Reference: https://ollama.com)

## Embeddings
Vector embeddings use transformer models to convert data like words, sentences, images, or documents into numerical vectors that represent their semantics. These vectors are points in a multidimensional space, where proximity indicates semantic similarity. 

(Reference: https://docs.oracle.com/en/database/oracle/machine-learning/oml4sql/23/mlsql/embedding.html#GUID-B8EC6F74-1517-4002-8E07-38F9B9B8CAA1)

## Vector databases
A vector database is a collection of data stored as mathematical representations. It is any database that can natively store and manage vector embeddings and handle the unstructured data they describe, such as documents, images, video, or audio. In this project we use FAISS as our vector database.

(Reference: https://www.oracle.com/database/vector-database/)

## Reranking
Reranking seeks to improve search relevance by reordering the result set returned by a search with a different model.
Reranking computes a relevance score between the query and each data object, and returns the list of objects sorted from the most to the least relevant.

(Reference: https://weaviate.io/developers/weaviate/concepts/reranking#:~:text=Reranking%20seeks%20to%20improve%20search,most%20to%20the%20least%20relevant.)

## Getting Started

### Directory setup
First make a dedicated folder "Research Ideas" on your Desktop
Copy the following files to the folder:
1) research_rag.py
2) cricket_faqs.txt (This is a default file, please replace it with your own)
3) requirements_for_project.txt

### LLM Setup
Run the following commands to pull models
For query- Use llama3.1 (ollama run llama3.1)
For embedding- Use nomic-embed-text (ollama run nomic-embed-text)

### Executing the code
1) Make a new conda environment with Python version 3.10 (conda create --name myenv python=3.10)
2) Activate the newly created environment (conda activate myenv)
3) In this environment run the following command (pip install -r requirements_for_project.txt)
4) To finally run the code, use the command (streamlit run research_rag.py), this shall re-direct you to the IP address where the code will run.
   (Note: Give the entire path to your file while running it)
