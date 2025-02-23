from models import SessionLocal, LocalChatSettings


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
def update_setting(
    system_prompt, llm, top_k_input, top_p_input, temperature, chat_token_limit
):
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
        else:
            raise Exception("Setting with ID 1 does not exist.")
    except Exception as e:
        db.rollback()
        raise Exception(f"Error: {str(e)}")
    finally:
        db.close()
