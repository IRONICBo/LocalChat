import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Load and process the document
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    # documents=all_splits,
    collection_name="default",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever()

# Define the prompt and LLM
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

ollama_llm = "qwen2:0.5b"
model_local = ChatOllama(model=ollama_llm)

# Define the RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_local
    | StrOutputParser()
)

def answer_question(question):
    try:
        # Run the question through the chain
        result = chain.invoke(question)
        return result
    except Exception as e:
        return f"An error occurred: {e}"

# Define the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# LangChain-Powered Q&A Interface")

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="Enter your question", placeholder="Type your question here...", lines=2)
            submit_button = gr.Button("Submit")

        with gr.Column():
            answer_output = gr.Textbox(label="Answer", placeholder="The answer will appear here...", lines=4)

    submit_button.click(answer_question, inputs=[question_input], outputs=[answer_output])

# Launch the interface
interface.launch()
