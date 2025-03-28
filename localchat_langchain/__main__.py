import json
import os
import sys
import re
import time
import logger

import gradio as gr
from openai import OpenAI
import requests
from langchain_retrival import (
    get_question_documents_with_collection,
    get_retrieved_documents,
    get_retrieved_documents_with_collection,
    retrival_tab,
    save_question_documents_with_collection,
)

# from zotero import zotero_manager_tab
from settings import settings_tab
from vectormanager import fetch_document_libraries

# from files import files_tab
# from monitor import monitor_tab
from manager import manager_tab
from models import DocumentLibrary, Session, SessionLocal, ChatbotUsage
from utils.abstract import get_abstract
from utils.alert import show_info, show_warning
from db.settings import fetch_setting
from settings import DEFAULT_RAG_PROMPT_TEMPLATE

log = logger.Logger("localchat.log")
sys.stdout = log


def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def convert_reference_text(content, link):
    abstract = get_abstract(content)
    return f"""
<details>
  <summary>{abstract}</summary>
  <p>{content}</p>
  <p><strong>For more details, visit <a href="{link}" target="_blank">this reference</a>.</strong></p>
</details>
"""


def convert_highlight_thinktext(text):
    """Convert <think> content to highlighted lines with > DeepThink:."""

    def format_think_content(content):
        lines = content.splitlines()
        # Line 1 with DeepThink label
        formatted_lines = [f"> DeepThink: {lines[0].strip()}"]
        # Line 2 and beyond
        for line in lines[1:]:
            formatted_lines.append(f"> {line.strip()}")
        return "\n".join(formatted_lines)

    new_text = re.sub(
        r"<think>(.*?)</think>",
        lambda match: f"{format_think_content(match.group(1))}",
        text,
        flags=re.DOTALL,
    )
    return new_text


def bot(
    history,
    model="qwen:0.5b",
    knowledge_base_choice=None,
    session_state=None,
    session_history_choice_id=None,
):
    system_prompt, _, top_k, top_p, temperature, max_tokens = fetch_setting()

    print(session_history_choice_id)

    if session_history_choice_id is None or session_history_choice_id == -1:
        if session_state is None:
            session_state = create_new_session()
        if isinstance(session_state, gr.State):
            session_state = session_state.value
    else:
        session_state = session_history_choice_id
        # TODO: move these logic to dropdown hooks
        history = get_session_history(session_state)
        history = json.loads(history)
        print("Session history loaded {session_state} with history {history}")
        show_info(f"Session {session_state} loaded.")

    if knowledge_base_choice is None:
        show_warning(
            "Currently, you have not selected any knowledge base. Please select a knowledge base to improve your chatting experience."
        )

    if history[-1][0] is None or model is None:
        return history, gr.MultimodalTextbox(value=None, interactive=False)

    # TODO: Add system prompt

    history[-1][1] = ""
    print(f"History: {history}")

    # get question
    question = history[-1][0]
    similarity_question = get_question_documents_with_collection(question)
    show_warning(f"Similarity question: {similarity_question}")

    save_question_documents_with_collection(question)
    print(f"Question: {question}")

    history_openai_format = []
    for human, assistant in history[:-1]:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})

    # RAG
    # Add retrival results to history
    knowledge_base_references = []
    # Disable knowledge base if knowledge_base_choice is None or -1
    if knowledge_base_choice is not None and knowledge_base_choice != -1:
        knowledge_base = get_retrieved_documents_with_collection(
            question, knowledge_base_choice
        )
        kb_data = ""
        for doc in knowledge_base:
            print(doc)
            kb_data += doc.page_content + "\n"

            # makesure this uuid key is exists
            knowledge_base_references.append((doc.page_content, doc.metadata["uuid"]))
        if kb_data != "":
            kb_data = (
                f"{DEFAULT_RAG_PROMPT_TEMPLATE}:{kb_data}, User Question is: {question}"
            )

        history_openai_format.append({"role": "user", "content": kb_data})
    else:
        history_openai_format.append({"role": "user", "content": history[-1][0]})

    print(f"Prompts: {history_openai_format}")

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://127.0.0.1:11434/v1",
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=history_openai_format,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )
        print(
            f"Chat with config model: {model} temperature: {temperature} top_p: {top_p} top_k: {top_k} max_tokens: {max_tokens}"
        )
    except Exception as e:
        print(f"Error: {e}")
        show_warning(f"Error: {e}")
        return history, gr.MultimodalTextbox(value=None, interactive=False)

    start_time = time.time()

    for chunk in completion:
        history[-1][1] += chunk.choices[0].delta.content

        # Output conversion, now for deepseek CoT
        if model.startswith("deepseek"):
            history[-1][1] = convert_highlight_thinktext(history[-1][1])

        yield history

    # Add references link
    if len(knowledge_base_references) != 0:
        for i, (content, uuid) in enumerate(knowledge_base_references):
            folder = uuid[:2]
            file_path = uuid[2:]
            history[-1][1] += "\n"
            history[-1][1] += convert_reference_text(
                content, f"http://127.0.0.1:8082/static/{folder}/{file_path}"
            )

    # Create session to save usage
    history_json = json.dumps(history)
    print(history_json)
    update_session_state(session_state, history_json)

    end_time = time.time()
    response_time = end_time - start_time

    try:
        # Calculate token count and tokens per second
        completion_without_stream = client.chat.completions.create(
            model=model,
            messages=history_openai_format,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
    except Exception as e:
        print(f"Error: {e}")
        show_warning(f"Error: {e}")
        return history, gr.MultimodalTextbox(value=None, interactive=False)

    total_token_count = completion_without_stream.usage.total_tokens
    prompt_tokens_count = completion_without_stream.usage.prompt_tokens
    completion_tokens_count = completion_without_stream.usage.completion_tokens

    print(f"Response time: {response_time:.2f}s")

    db = SessionLocal()
    try:
        usage_record = ChatbotUsage(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            total_token_count=total_token_count,
            completion_tokens_count=completion_tokens_count,
            prompt_tokens_count=prompt_tokens_count,
            response_time=response_time,
        )
        db.add(usage_record)
        db.commit()
    except Exception as e:
        print(f"Failed to insert data into database: {e}")
        db.rollback()
    finally:
        db.close()

    # TODO: Current state is not valid in multi chat
    yield history


def fetch_model_names():
    """Fetch model names from the API."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        model_names = [model["name"] for model in models]
        return model_names
    except Exception as e:
        return ["Error fetching models"]


# Helper function to fetch all document libraries
def fetch_document_pairs():
    db = SessionLocal()
    try:
        libraries = db.query(DocumentLibrary).all()
        data = [(lib.name, lib.id) for lib in libraries]
        # Add disable one
        data.append(("Disable", -1))
        return data
    finally:
        db.close()


def update_model_dropdown():
    """Update Dropdown with model names."""
    model_names = fetch_model_names()
    knowledge_base_names = fetch_document_pairs()
    session_histories = fetch_session_history_pairs()
    print(knowledge_base_names)
    return (
        gr.update(choices=model_names, value=None),
        gr.update(choices=knowledge_base_names, value=None),
        gr.update(choices=session_histories, value=-1),
    )


# Helper function to fetch all dialogue history pairs (for dropdown etc.)
def fetch_session_history_pairs():
    db = SessionLocal()
    try:
        sessions = db.query(Session).order_by(Session.created_at.desc()).all()
        data = [
            (
                f"{session.description or 'No description'} (ID: {session.id})",
                session.id,
            )
            for session in sessions
        ]
        data.insert(0, ("Disable", -1))
        return data
    finally:
        db.close()


def clear_history():
    return [], gr.MultimodalTextbox(value=None, interactive=False)


def create_session_state():
    session_id = create_new_session()
    return gr.State(session_id)


def create_new_session():
    db = SessionLocal()
    try:
        # Create a new session
        session = Session(
            description="",
            history="",
            llm="",
            llm_settings="",
            similarity_threshold=0.5,
            vector_similarity_weight=0.5,
        )
        db.add(session)
        db.commit()
        return session.id
    finally:
        db.close()


# Update session state
# In current senario, we just need to cover all of the history to current session
# TODO: support save session config
def update_session_state(session_id, session_history):
    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if session and session_history:
            session.history = session_history
            db.commit()
            return session.id
        else:
            return None
    finally:
        db.close()


def get_session_history(session_id):
    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if session:
            return session.history
        else:
            return None
    finally:
        db.close()


def chat_tab():
    chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False)
    # Create a new state variable to store the chat history
    session_id = create_new_session()
    session_state = gr.State(session_id)

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=[],
        placeholder="Enter message or upload file...",
        show_label=False,
    )

    # Disable now
    # tag_selector = gr.Radio(
    #     choices=["Loading..."],
    #     label="Choose a Tag",
    #     interactive=True,
    # )

    with gr.Row():
        with gr.Column(scale=3):
            model_names = fetch_model_names()
            model_choice = gr.Dropdown(
                choices=model_names,
                value=model_names[0],
                label="Choose Model",
            )

            document_pairs = fetch_document_pairs()
            knowledge_base_choice = gr.Dropdown(
                choices=document_pairs,
                value=None,
                label="Choose Knowledge Base",
            )

            session_history_pairs = fetch_session_history_pairs()
            session_history_choice = gr.Dropdown(
                choices=session_history_pairs,
                value=None,
                label="Choose Session",
            )

        with gr.Column(scale=1):
            update_button = gr.Button("Refresh Config")
            update_button.click(
                fn=update_model_dropdown,
                inputs=[],
                outputs=[model_choice, knowledge_base_choice],
            )

            clear = gr.ClearButton([chat_input, chatbot], value="Clear History")
            clear.click(fn=create_session_state, inputs=[], outputs=[session_state])

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(
        bot,
        [
            chatbot,
            model_choice,
            knowledge_base_choice,
            session_state,
            session_history_choice,
        ],
        [chatbot],
        api_name="bot_response",
    )
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])


with gr.Blocks() as main_block:
    gr.Markdown(
        """
        <h1><center>🚀🚀🚀 LocalChat 🚀🚀🚀</center></h1>
        <p><center>LocalChat is designed for personal AI chatbot that uses the private LLM models with knowledge base support.</center></p>
    """
    )

    with gr.Tabs():
        with gr.Tab(label="Chat"):
            chat_tab()

        with gr.Tab(label="Manager"):
            manager_tab()

        # with gr.Tab(label="Monitor"):
        # monitor_tab()

        with gr.Tab(label="Retrival"):
            retrival_tab()

        # with gr.Tab(label="Zotero Helper"):
        #     zotero_manager_tab()

        with gr.Tab(label="Settings"):
            settings_tab()

main_block.queue()
main_block.launch()
