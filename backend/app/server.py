from operator import itemgetter
from typing import Any
import time

from pydantic import BaseModel

from decouple import config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.globals import set_llm_cache
from langchain.schema import Generation
from langchain_redis import RedisSemanticCache
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

import redis as redis

REDIS_URL = config("REDIS_URL")
collection_name = config("COLLECTION_NAME")
url = config("QDRANT_URL")
llm_model = config("LLM_MODEL")
embedding_model = config("EMBEDDING_MODEL")
redis_host = config("REDIS_HOST")
redis_port = config("REDIS_PORT")
embedding_dimension = config("EMBEDDING_DIMENSION")
ollama_base_url = config("OLLAMA_BASE_URL")

class Input(BaseModel):
    query: str

app = FastAPI()
# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
print("Before LLM Initialization")
# LLM Model
llm = OllamaLLM(
    model=llm_model,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    base_url=ollama_base_url,
)
print("After LLM Initialization")

# Embeddings Model
embeddings = OllamaEmbeddings(
    model=embedding_model,
    base_url=ollama_base_url,
)
print("After Embedding model Initialization")

# Prompt Template
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# Connecting to redis
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
print("Connection Done!")

# Semantic Cache
print(REDIS_URL)
semantic_cache = RedisSemanticCache(
    redis_url=REDIS_URL, embeddings=embeddings, distance_threshold=0.3
)
print("After RedisSemanticCache Initialization")

set_llm_cache(semantic_cache)
print("After set_llm_cache(semantic_cache)")

# Vector Store Qdrant Running Locally on docker container

def create_collection():
    client = QdrantClient(url=url)

    if not client.collection_exists(collection_name=collection_name):
        print(
            f"Collection '{collection_name}' does not exist. Creating new collection."
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dimension, distance=Distance.COSINE
            ),
        )

    return client


client = create_collection()
qdrant_vectorstore = QdrantVectorStore(
    client=client, collection_name=collection_name, embedding=embeddings
)

retriever = qdrant_vectorstore.as_retriever()

rag_chain = (
    {"context": itemgetter("query") | retriever, "query": itemgetter("query")}
    | rag_prompt
    | llm
)

# Define a new route to measure response time
@app.post("/query_with_timing")
async def query_with_timing(input: Input):
    start_time = time.time()  # Start the timer

    # Process the input with rag_chain
    print("input.query : " + input.query)
    response = rag_chain.invoke({"query": input.query})
    end_time = time.time()  # End the timer
    response_time = end_time - start_time  # Calculate the time taken

    return JSONResponse(content={"response": response, "response_time": response_time})


# Add the existing chain route using langserve
add_routes(
    app,
    rag_chain.with_types(input_type=Input, output_type=str).with_config(
        {"run_name": "AIBillOfRights"}
    ),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
