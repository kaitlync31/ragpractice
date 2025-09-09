import os

import chromadb
import openai
import nltk
from chromadb.utils import embedding_functions
from nltk import sent_tokenize

nltk.download('punkt_tab')

# Initialize clients
chroma_client = chromadb.Client()
openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

# Create a new collection
collection = chroma_client.get_or_create_collection(name="rag_practice")

# Load document
doc = open("docs/chapter31.md").read()

# Chunk document into smaller pieces
chunked_data = sent_tokenize(doc)

# Embed the chunks
embedding_function = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
embeddings = embedding_function(chunked_data)

# Store embeddings in ChromaDB
collection.add(documents=chunked_data, embeddings=embeddings, ids = [f"id.{i}" for i in range(len(chunked_data))])

# Create a system prompt template
SYSTEM_PROMPT = """
 
You are an expert in AP US History. You will provide a response 
to a student query using exact language from the provided relevant chunks of text.
 
RELEVANT CHUNKS:
 
{relevant_chunks}
 
"""

# Get user query
user_message = "What social changes happened in America in the 20s?"
print("User message:", user_message)

# Get relevant documents from ChromaDB
relevant_chunks = collection.query(query_embeddings = embedding_function([user_message]), n_results = 2)['documents'][0]
print("Retrieved Chunks: " + str(relevant_chunks))

# Send query and relevant documents to OpenAI
system_prompt = SYSTEM_PROMPT.format(relevant_chunks = "\n".join(relevant_chunks))
response = openai_client.chat.completions.create(model="gpt-4-turbo-preview", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
print("RAG-GPT Response: " + response.choices[0].message.content)
