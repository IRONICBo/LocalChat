from llama_index import LLMPredictor, ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import Ollama
import chromadb

db_path = "./chroma_db"
db_client = chromadb.PersistentClient(path=db_path)
collection_name = "default"
chroma_collection = db_client.get_or_create_collection(collection_name)

# Initialize embedding model and Ollama LLM
embed_args = {
    "model_name": "maidalun1020/bce-embedding-base_v1",
    "max_length": 512,
    "embed_batch_size": 32,
    "device": "cpu",
}
embed_model = HuggingFaceEmbedding(**embed_args)
llm = Ollama(model="qwen2:0.5b")  # Your Ollama LLM instance


documents = SimpleDirectoryReader(".").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=None), embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model, show_progress=True, llm=None,
    service_context=service_context,
)

query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
response = query_engine.query("What did the author do growing up?")

print(response.response)