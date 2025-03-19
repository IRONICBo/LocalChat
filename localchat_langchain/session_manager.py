import json
import gradio as gr
from sqlalchemy.orm import Session as DBSession
from model_manager import DEFAULT_PAGE_NUM, DEFAULT_PAGE_SIZE
from models import Session, get_db

# Helper function to fetch all session records
def fetch_session_pairs():
    db = next(get_db())
    try:
        sessions = db.query(Session).all()
        data = [(f"Desc: {session.description} ID: {session.id}", session.id) for session in sessions]
        return data
    finally:
        db.close()

def get_session_history(session_id):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if session:
            return session
        else:
            return None
    finally:
        db.close()

def fetch_session_list(page_number_input, page_size_input, session_id):
    """
    collection_name is the document name in chroma db.
    source is the generated filename for specific file.
    """
    db = next(get_db())
    current_session = get_session_history(session_id)
    history = current_session.history
    history = json.loads(history)
    print(history)

    chunks = [
        [
            current_session.id,
            current_session.description,
            # TODO: fix with role
            his[0], # user
            his[1], # content
            current_session.llm,
            current_session.llm_settings,
            current_session.similarity_threshold,
            current_session.vector_similarity_weight,
            current_session.created_at,
        ]
        for his in history
    ]

    return chunks


# Update session details function
def update_session(session_id, description, llm, llm_settings, similarity_threshold, vector_similarity_weight, history):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if session:
            session.description = description
            session.llm = llm
            session.llm_settings = llm_settings
            session.similarity_threshold = similarity_threshold
            session.vector_similarity_weight = vector_similarity_weight
            session.history = history
            db.commit()
            return f"Session {session_id} updated successfully!"
        else:
            return f"Session {session_id} not found!"
    finally:
        db.close()

# Session management UI function
def session_manager_tab():
    gr.Markdown("## Manage Sessions")

    with gr.Row():
        with gr.Column(scale=1):
            sessions = fetch_session_pairs()
            print(sessions)
            session_choice = gr.Dropdown(
                choices=sessions,
                label="Choose Session",
            )
            page_number_input = gr.Number(
                label="Page Number", value=DEFAULT_PAGE_NUM, precision=0
            )
            page_size_input = gr.Slider(
                label="Page Size",
                value=DEFAULT_PAGE_SIZE,
                minimum=1,
                maximum=10,
                step=1,
            )
            fetch_files_metadata_button = gr.Button("Refresh Session Query Param")


        with gr.Column(scale=3):
            session_list = gr.Dataframe(
                label="Session",
                headers=[
                    "ID",
                    "Description",
                    "Role",
                    "Content",
                    "LLM",
                    "LLM Settings",
                    "Similarity Threshold",
                    "Vector Similarity Weight",
                    "Created At",
                ],  # Specify the headers
                interactive=True,
            )

        fetch_files_metadata_button.click(
            fn=fetch_session_list,
            inputs=[page_number_input, page_size_input, session_choice],
            outputs=session_list,
        )

# Main Gradio app
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>Session Manager</center></h1>")
        session_manager_tab()

    main_block.queue()
    main_block.launch()
