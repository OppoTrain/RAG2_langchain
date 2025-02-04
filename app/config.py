from dotenv import load_dotenv
import os
from together import Together
from transformers import GPT2Tokenizer
from langchain_huggingface import HuggingFaceEmbeddings  
import chromadb

# Load environment variables
load_dotenv()

# Retrieve paths & API keys
CHROMADB_PATH = os.getenv("CHROMADB_PATH")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Validate environment variables
if not CHROMADB_PATH:
    raise ValueError("Error: CHROMADB_PATH is not set in the environment variables.")
if not TOGETHER_API_KEY:
    raise ValueError("Error: TOGETHER_API_KEY is not set in the environment variables.")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMADB_PATH)

# Load tokenizer & embedding function
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
hf_embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Initialize Together API client
together_client = Together(api_key=TOGETHER_API_KEY)

# Function to get or create a ChromaDB collection
def get_or_create_collection(client, collection_name):
    try:
        return client.get_collection(collection_name)  # Try to get the existing collection
    except Exception:
        print(f"Creating new collection: {collection_name}")
        return client.create_collection(collection_name)  # Create if not found

# Load the main RAG data collection
collection = get_or_create_collection(client, "RAG_Data")

# Get or create the QA collection
qa_collection = get_or_create_collection(client, "qa_collection")

print("Collections loaded successfully.")
