import gradio as gr
import os

from models import SessionLocal, LocalChatSettings
from utils.alert import show_info, show_warning
from utils.model_helper import fetch_model_names

# Helper function to fetch the setting with id=1
def fetch_setting():
    db = SessionLocal()
    try:
        setting = db.query(LocalChatSettings).filter(LocalChatSettings.id == 1).first()
        if setting:
            return [
                    setting.system_prompt,
                    setting.llm,
                    setting.top_k,
                    setting.top_p,
                    setting.temperature,
                    setting.chat_token_limit,
                ]
        else:
            return "Setting with ID 1 does not exist."
    finally:
        db.close()


# Helper function to update the setting with id=1
def update_setting(system_prompt, llm, top_k_input, top_p_input, temperature, chat_token_limit):
    db = SessionLocal()
    try:
        setting = db.query(LocalChatSettings).filter(LocalChatSettings.id == 1).first()
        if setting:
            setting.system_prompt = system_prompt
            setting.llm = llm
            setting.temperature = temperature
            setting.top_k = top_k_input
            setting.top_p = top_p_input
            setting.chat_token_limit = chat_token_limit
            db.commit()
            show_info("Setting with ID 1 has been updated successfully.")
        else:
            show_warning("Setting with ID 1 does not exist.")
    except Exception as e:
        db.rollback()
        show_warning(f"Error: {str(e)}")
    finally:
        db.close()

    return fetch_setting()

# Gradio UI for querying and updating the setting with id=1
def settings_tab():
    gr.Markdown("## Manage Settings")

    system_prompt_default, llm_default, top_k_default, top_p_default, temperature_default, max_tokens_default = fetch_setting()
    model_names = fetch_model_names()

    # Update Setting Section
    gr.Markdown("##### Static Setting")
    # TODO: Add Ollama API and File Root Path in db
    gr.Textbox(label="Ollama API", value="https://localhost:11434")
    gr.Textbox(label="File Root Path", value=os.path.join(os.getcwd(), ".files"))

    gr.Markdown("##### Modify Setting")
    system_prompt_input = gr.Textbox(label="System Prompt", value=system_prompt_default)
    llm_input = gr.Dropdown(label="LLM", value=llm_default, choices=model_names)
    top_k_input = gr.Slider(label="Top K", value=top_k_default, minimum=0.0, maximum=1.0, step=0.01)
    top_p_input = gr.Slider(label="Top P", value=top_p_default, minimum=0.0, maximum=1.0, step=0.01)
    temperature_input = gr.Slider(label="Temperature", value=temperature_default, minimum=0.0, maximum=1.0, step=0.01)
    max_tokens_input = gr.Slider(label="Max Tokens", value=max_tokens_default, minimum=32, maximum=8192, step=32)
    update_button = gr.Button("Update Setting")

    # Update the setting with id=1
    update_button.click(
        fn=update_setting,
        inputs=[
            system_prompt_input,
            llm_input,
            top_k_input,
            top_p_input,
            temperature_input,
            max_tokens_input,
        ],
    )


# Main Gradio app
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>LocalChatRagSettings Manager (ID=1)</center></h1>")
        with gr.Tabs():
            with gr.Tab(label="Setting Manager"):
                settings_tab()

    main_block.queue()
    main_block.launch()
