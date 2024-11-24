import gradio as gr
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import Ollama
from llama_index.retrievers import VectorIndexRetriever
from BCEmbedding.tools.llama_index import BCERerank

# Initialize embedding model and reranker model
embed_args = {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length': 512, 'embed_batch_size': 32, 'device': 'cpu'}
embed_model = HuggingFaceEmbedding(**embed_args)

reranker_args = {'model': 'maidalun1020/bce-reranker-base_v1', 'top_n': 5, 'device': 'cpu'}
reranker_model = BCERerank(**reranker_args)

llm = Ollama(model="qwen2:0.5b")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Create a global index for the knowledge base
global_index = None

# Function to upload and index documents into the knowledge base
def upload_and_index(file):
    global global_index
    # Load document content
    documents = SimpleDirectoryReader(input_files=[file.name]).load_data()
    # Parse document into nodes
    node_parser = SimpleNodeParser.from_defaults(chunk_size=400, chunk_overlap=80)
    nodes = node_parser.get_nodes_from_documents(documents)
    # Create or update the knowledge base index
    if global_index is None:
        global_index = VectorStoreIndex(nodes, service_context=service_context)
    else:
        global_index.insert_nodes(nodes)
    return "Document successfully uploaded and added to the knowledge base!"

# Function to retrieve information and answer a question
def answer_question(query):
    if global_index is None:
        return "The knowledge base is empty. Please upload a document first!"
    # Retrieve relevant content from the knowledge base
    vector_retriever = VectorIndexRetriever(index=global_index, similarity_top_k=10, service_context=service_context)
    retrieval_by_embedding = vector_retriever.retrieve(query)
    retrieval_by_reranker = reranker_model.postprocess_nodes(retrieval_by_embedding, query_str=query)
    # Generate a response using the language model
    query_engine = global_index.as_query_engine(node_postprocessors=[reranker_model])
    query_response = query_engine.query(query)
    return f"Retrieved Content:\n{retrieval_by_reranker[0].text}\n\n Retrieved Score: {retrieval_by_reranker[0].score}\n\n Model Answer:\n{query_response}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Document Knowledge Base and Q&A System")
    gr.Markdown("## Features: Document Upload, Knowledge Base Construction, and Q&A Generation")

    # Tab for document upload
    with gr.Tab("Upload Document"):
        file_input = gr.File(label="Upload Document", file_types=[".txt", ".md", ".pdf", ".py"])
        upload_button = gr.Button("Upload and Add to Knowledge Base")
        upload_status = gr.Textbox(label="Status", interactive=False)
        upload_button.click(upload_and_index, inputs=[file_input], outputs=[upload_status])

    # Tab for asking questions
    with gr.Tab("Ask Question"):
        question_input = gr.Textbox(label="Enter Your Question")
        answer_output = gr.Textbox(label="Answer", interactive=False)
        ask_button = gr.Button("Get Answer")
        ask_button.click(answer_question, inputs=[question_input], outputs=[answer_output])

# Launch the Gradio app
demo.launch()
