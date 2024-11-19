import gradio as gr
from BCEmbedding.tools.llama_index import BCERerank
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.retrievers import VectorIndexRetriever
from llama_index.llms import OpenAI
import os

# Initialize embedding model and reranker model
embed_args = {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length': 512, 'embed_batch_size': 32, 'device': 'cpu'}
embed_model = HuggingFaceEmbedding(**embed_args)

reranker_args = {'model': 'maidalun1020/bce-reranker-base_v1', 'top_n': 5, 'device': 'cpu'}
reranker_model = BCERerank(**reranker_args)

# Initialize LLM
llm = OpenAI(model='gpt-3.5-turbo-0613', api_key=os.environ.get('OPENAI_API_KEY'), api_base=os.environ.get('OPENAI_BASE_URL'))
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Function to process uploaded files and execute query
def process_file_and_query(file, prompt):
    try:
        # Load document from the uploaded file
        documents = SimpleDirectoryReader(input_files=[file.name]).load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=400, chunk_overlap=80)
        nodes = node_parser.get_nodes_from_documents(documents)

        # Build index
        index = VectorStoreIndex(nodes, service_context=service_context)

        # Retrieve and rerank results
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10, service_context=service_context)
        retrieval_by_embedding = vector_retriever.retrieve(prompt)
        retrieval_by_reranker = reranker_model.postprocess_nodes(retrieval_by_embedding, query_str=prompt)

        # Prepare response
        results = [{"content": node.get_text(), "score": node.get_score()} for node in retrieval_by_reranker]
        return results
    except Exception as e:
        return str(e)

# Gradio Interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Document Retrieval Dashboard")
        gr.Markdown("Upload a document, enter your prompt, and get the most relevant results.")

        with gr.Row():
            file_input = gr.File(label="Upload Document")
            prompt_input = gr.Textbox(label="Enter your prompt")

        output = gr.Dataframe(label="Retrieval Results", headers=["Content", "Score"], interactive=False)

        submit_btn = gr.Button("Retrieve")
        submit_btn.click(process_file_and_query, inputs=[file_input, prompt_input], outputs=output)

    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
