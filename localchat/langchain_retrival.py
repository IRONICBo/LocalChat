import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from uuid import uuid4
import os

# File upload directory
UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Function to process uploaded file and add to vectorstore
def process_file(file):
    file_path = os.path.join(UPLOAD_DIRECTORY, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split the content into documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_text(content)

    # Convert to LangChain Documents with metadata
    langchain_documents = [
        Document(page_content=doc, metadata={"source": file.name}) for doc in documents
    ]

    # Generate UUIDs for the documents
    uuids = [str(uuid4()) for _ in langchain_documents]

    # Add documents to vectorstore
    vectorstore.add_documents(documents=langchain_documents, ids=uuids)
    return "File processed and added to vectorstore."

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    collection_name="default",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever(search_type="similarity", k=2)

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
        print("Input Question:", question)  # Print the input question
        result = chain.invoke(question)  # Retrieve chain results
        return result
    except Exception as e:
        return f"An error occurred: {e}"

# Define the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# LangChain-Powered File Upload Q&A Interface")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload a text file", file_types=[".txt"], type="file")
            process_button = gr.Button("Process File")
            process_output = gr.Textbox(label="File Processing Status", interactive=False)

            question_input = gr.Textbox(label="Enter your question", placeholder="Type your question here...", lines=2)
            submit_button = gr.Button("Submit")

        with gr.Column():
            answer_output = gr.Textbox(label="Answer", placeholder="The answer will appear here...", lines=4)

    def process_and_respond(file, question):
        status = process_file(file) if file else "No file uploaded."
        if question:
            answer = answer_question(question)
        else:
            answer = "No question provided."
        return status, answer

    process_button.click(process_file, inputs=[file_input], outputs=[process_output])
    submit_button.click(answer_question, inputs=[question_input], outputs=[answer_output])

# Launch the interface
interface.launch()
