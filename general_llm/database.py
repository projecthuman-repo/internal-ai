from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API")
pinecone_api_key = os.getenv("PINECONE_API")
openai_key=os.getenv("OPENAI_API_KEY")

# --------------------Parsing and loading the data------------------------------------------------
parser = LlamaParse(
    api_key=llama_cloud_api_key,
    result_type="markdown",  # "markdown" and "text" are available
)
file_extractor = {".pdf": parser}
reader = SimpleDirectoryReader("./data", file_extractor=file_extractor)
documents = reader.load_data()

# Initializing  vector embedding model
embed_model = OpenAIEmbedding(api_key=openai_key)



# -------------Pinecone initialization-----------------------#

# Initialize connection to Pinecone
pc = PineconeGRPC(api_key=pinecone_api_key)
index_name = "phc-data"

# Create your index (can skip this step if your index already exists)
if index_name not in pc.list_indexes().names():

    pc.create_index(
        index_name,
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Initialize your index
pinecone_index = pc.Index(index_name)

# Initialize VectorStore
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Our pipeline with the addition of our PineconeVectorStore
pipeline = IngestionPipeline(
    transformations=[
        SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95, 
            embed_model=embed_model,
            ),
        embed_model,
        ],
        vector_store=vector_store  # Our vector database
    )

# Now we run our pipeline
pipeline.run(documents=documents)
pinecone_index.describe_index_stats()
