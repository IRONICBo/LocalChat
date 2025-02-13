import gradio as gr
from models import SessionLocal, LocalChatSettings

# Helper function to fetch the setting with id=1
def fetch_setting():
    db = SessionLocal()
    try:
        setting = db.query(LocalChatSettings).filter(LocalChatSettings.id == 1).first()
        if setting:
            return {
                "System Prompt": setting.system_prompt,
                "LLM": setting.llm,
                "Temperature": setting.temperature,
                # "Max Tokens": setting.chat_token_limit,
                "Chat Token Limit": setting.chat_token_limit
            }
        else:
            return "Setting with ID 1 does not exist."
    finally:
        db.close()

# Helper function to update the setting with id=1
def update_setting(system_prompt, llm, temperature, chat_token_limit):
    db = SessionLocal()
    try:
        setting = db.query(LocalChatSettings).filter(LocalChatSettings.id == 1).first()
        if setting:
            setting.system_prompt = system_prompt
            setting.llm = llm
            setting.temperature = temperature
            setting.chat_token_limit = chat_token_limit
            db.commit()
            return "Setting with ID 1 has been updated successfully."
        else:
            return "Setting with ID 1 does not exist."
    except Exception as e:
        db.rollback()
        return f"Error: {str(e)}"
    finally:
        db.close()

# Gradio UI for querying and updating the setting with id=1
def settings_tab():
    gr.Markdown("## Manage Setting")

    # Query Setting Section
    query_button = gr.Button("Query Setting")
    query_result = gr.Textbox(label="Current Setting (ID=1)", interactive=False, lines=5)

    # Update Setting Section
    system_prompt_input = gr.Textbox(label="System Prompt")
    llm_input = gr.Textbox(label="LLM")
    temperature_input = gr.Number(label="New Temperature", value=0.1)
    # max_tokens_input = gr.Number(label="New Max Tokens", value=4000)
    chat_token_limit_input = gr.Number(label="New Chat Token Limit", value=4000)
    update_button = gr.Button("Update Setting")
    update_result = gr.Textbox(label="Update Result", interactive=False)

    # Fill the inputs with the current setting values after query
    def update_inputs():
        db = SessionLocal()
        try:
            setting = db.query(LocalChatSettings).filter(LocalChatSettings.id == 1).first()
            if setting:
                return [
                    setting.system_prompt,
                    setting.llm,
                    setting.temperature,
                    setting.chat_token_limit
                ]
            else:
                return "Setting with ID 1 does not exist."
        finally:
            db.close()

    # Button Click Events
    # Query current setting with id=1
    query_button.click(fn=fetch_setting, outputs=query_result)
    query_button.click(fn=update_inputs, outputs=[system_prompt_input, llm_input, temperature_input, chat_token_limit_input])

    # Update the setting with id=1
    update_button.click(
        fn=update_setting,
        inputs=[system_prompt_input, llm_input, temperature_input, chat_token_limit_input],
        outputs=update_result
    )

# Main Gradio app
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>LocalChatRagSettings Manager (ID=1)</center></h1>")
        with gr.Tabs():
            with gr.Tab(label="Setting Manager"):
                setting_manager_tab()

    main_block.queue()
    main_block.launch()
