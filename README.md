# General LLM

## RAG-Based Chatbot

A chatbot leveraging Retrieval-Augmented Generation (RAG) to answer queries based on company data.

## Summary

This chatbot utilizes LLamaParse to parse data for improved results, converting it into vector embeddings using OpenAI’s embedding model. The embeddings are then stored in the Pinecone Vector Database. When a user submits a query, it is converted into a vector embedding. The retriever fetches the most relevant data from the database based on similarity, and this data, along with the query, is passed to the LLM to generate a response.

## Technology Used

- **Pinecone Vector DB** - API Key Required
- **LlamaCloud/LLamaParse** for file parsing - API Key Required
- **OpenAI Embedding Model** - API Key Required
- **LLamaIndex**

## Installation

1. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Set up your API keys in the `.env` file as follows:
    ```env
    LLAMA_CLOUD_API="YOUR_KEY_HERE"
    PINECONE_API="YOUR_KEY_HERE"
    OPENAI_API_KEY="YOUR_KEY_HERE"
    ```

3. Resolve any import errors by installing the required packages individually:
    ```bash
    pip install <package-name>
    ```

## Usage

### Upserting Data

1. Place your PDF documents in the `data` directory.(Llamaparse also supports ppt,images,spreadsheet,changes to the database.py should be made to achieve it)
2. Run the following command to upsert data into the vector database:
    ```bash
    python database.py
    ```
    **Note:** Run `database.py` only when you need to upsert new data.

### Running the Chatbot

Invoke the chatbot using the following command:
```bash
python llm.py
