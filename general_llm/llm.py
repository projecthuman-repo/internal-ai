from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import resolve_embed_model
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine

import os

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API")
openai_key = os.getenv("OPENAI_API_KEY")

embed_model = OpenAIEmbedding(api_key=openai_key)
pc = PineconeGRPC(api_key=pinecone_api_key)
index_name = "phc-data"
pinecone_index = pc.Index(index_name)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)


vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


# React Agent Initialization
llm = Ollama(model="llama2", request_timeout=1000000)
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriever)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="phc_data",
            description="this has data for the company project human city which encompasses three distinct applications: Lotus Learning, SpotStitch, and Co-Quest",
        ),
    ),
]

#Initializing a chatbot
chat_engine = vector_index.as_chat_engine(
    chat_mode="react",#using react,other
    llm=llm,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " You also act as an helpful assistant for the users of the company Project:Human City using relavant tools."
        "\nInstruction: Use the previous chat history,to interact and help the user."
    ),
    verbose=True,
)
print("You can start the conversation. Type 'exit', 'quit', or 'stop' to end the chat.")
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "stop"]:
        print("Exiting chat.")
        break
    response = chat_engine.stream_chat(user_input)
    for token in response.response_gen:
        print(token, end="")
    print("\n")
