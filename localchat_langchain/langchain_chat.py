import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# --- Embedding and Vector Store Setup ---

# Initialize the embedding model for text representation
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Create a persistent vector store using Chroma for similarity search
vectorstore = Chroma(
    collection_name="default",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Configure the retriever to fetch top-k similar documents
retriever = vectorstore.as_retriever(search_type="similarity", k=2)

# --- Add Example Document to Vector Store ---

# Create a sample document about the Leaning Tower of Pisa (in Chinese)
document_10 = Document(
    page_content=(
        "数据比萨斜塔从地基到塔顶高58.36米，从地面到塔顶高55米，钟楼墙体在地面上的宽度是5.09米，"
        "在塔顶宽2.48米，总重约14453吨，重心在地基上方22.6米处。圆形地基面积为285平方米，"
        "对地面的平均压强为497千帕。2010年时倾斜角度为3.97度，偏离地基外沿2.3米，顶层突出4.5米。"
    ),
    metadata={"source": "tweet"},
    id=10,
)

# Add the document to the vector store with a unique ID
documents = [document_10]
uuids = [str(uuid4()) for _ in documents]
vectorstore.add_documents(documents=documents, ids=uuids)

# --- Test Similarity Search ---

# Perform a similarity search for a sample question
results = vectorstore.similarity_search("比萨斜塔多高？", k=2)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

# --- Prompt and LLM Setup ---

# Define the prompt template for the LLM (RAG style)
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the local chat model (Ollama)
ollama_llm = "qwen2:0.5b"
model_local = ChatOllama(model=ollama_llm)

# --- RAG Chain Construction ---

# Compose the retrieval-augmented generation (RAG) chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_local
    | StrOutputParser()
)

# --- Question Answering Function ---

def answer_question(question):
    """
    Given a user question, retrieve relevant context and generate an answer using the RAG chain.
    Returns the answer or an error message.
    """
    try:
        print("Input Question:", question)  # Log the input question
        # Run the question through the chain and get the result
        result = chain.invoke(question)
        print("Model Output:", result)  # Log the model output
        return result
    except Exception as e:
        return f"An error occurred: {e}"

# --- Gradio Interface Definition ---

with gr.Blocks() as interface:
    gr.Markdown("# LangChain-Powered Q&A Interface")

    with gr.Row():
        with gr.Column():
            # Textbox for user to input their question
            question_input = gr.Textbox(
                label="Enter your question",
                placeholder="Type your question here...",
                lines=2,
            )
            # Button to submit the question
            submit_button = gr.Button("Submit")

        with gr.Column():
            # Textbox to display the answer from the model
            answer_output = gr.Textbox(
                label="Answer",
                placeholder="The answer will appear here...",
                lines=4
            )

    # Bind the submit button to the answer_question function
    submit_button.click(
        answer_question, inputs=[question_input], outputs=[answer_output]
    )

# --- Launch the Gradio Web Interface ---
interface.launch()
