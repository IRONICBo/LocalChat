import chromadb
# import gradio as gr
from llama_index.legacy import QueryBundle
from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# version issue
# Plain typing.TypeAlias is not valid as type argument

# Initialize local LLM and embedding model
# Define file upload directory
upload_path = "uploads"
ollama_embedding = OllamaEmbedding(model_name="nomic-embed-text:latest")
llm = Ollama(model="qwen2:0.5b")
service_context = ServiceContext.from_defaults(llm=None, embed_model=ollama_embedding)

db_path = "./chroma_db"
db_client = chromadb.PersistentClient(path=db_path)
collection_name = "default"
chroma_collection = db_client.get_or_create_collection(collection_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Load the vector store index
index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
# query_engine = index.as_query_engine(local_llm, similarity_top_k=10)
# retriever = index.as_retriever()
query_engine = index.as_query_engine(llm=llm, similarity_top_k=10)
query="Give me summary of water related issues"
bundle = QueryBundle(query, embedding=ollama_embedding.get_query_embedding(query))
result = query_engine.query(bundle)
# nodes = retriever.retrieve("apples")
# filters = {
#     "where": {"field_name": {"$exists": True}}  # 或者一些通用的条件
# }
# retriever.retrieve("apples")

# print(nodes)
print("Done")

# def retrieve_and_query(user_query):
#     """Function to process user query, retrieve relevant data, and provide information."""
#     try:
#         # Create query bundle with embedding
#         bundle = QueryBundle(user_query, embedding=ollama_embedding.get_query_embedding(user_query))

#         # Perform query
#         result = query_engine.query(bundle)

#         # Extract response information
#         response_text = result.response  # Query response (summary or answer)
#         sources = result.sources  # Related original data sources

#         source_info = "\n".join([f"Source {i+1}: {source}" for i, source in enumerate(sources)])

#         return response_text, source_info
#     except Exception as e:
#         return f"Error: {str(e)}", ""

# def retriever_tab():
#     gr.Markdown("## Retriever with Embedding Query")

#     query_input = gr.Textbox(label="Your Query", placeholder="Enter your query here")

#     response_output = gr.Textbox(label="Query Response", lines=5, interactive=False)
#     source_output = gr.Textbox(label="Related Sources", lines=5, interactive=False)

#     query_button = gr.Button("Submit")

#     query_button.click(
#         retrieve_and_query,
#         inputs=[query_input],
#         outputs=[response_output, source_output]
#     )

#     return query_input, response_output, source_output, query_button

# if __name__ == "__main__":
#     with gr.Blocks() as main_block:
#         gr.Markdown("<h1><center>Retriever with Embedding Query</center></h1>")

#         with gr.Tabs():
#             with gr.Tab(label="Retriever"):
#                 retriever_tab()

#     main_block.queue()
#     main_block.launch()
