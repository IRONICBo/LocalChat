import os
import sys
import time

import gradio as gr
from openai import OpenAI

from files import files_tab

# from monitor import monitor_tab
from manager import manager_tab
import logger
from models import SessionLocal, ChatbotUsage

log = logger.Logger("localchat.log")
sys.stdout = log


def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot(history, model="qwen:0.5b", temperature=0.1, max_tokens=1024):
    history[-1][1] = ""

    history_openai_format = []
    for human, assistant in history[:-1]:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
        # history_openai_format.append({"role": "assistant", "content": "数据比萨斜塔从地基到塔顶高58.36米，从地面到塔顶高55米，钟楼墙体在地面上的宽度是5.09米，在塔顶宽2.48米，总重约14453吨，重心在地基上方22.6米处。圆形地基面积为285平方米，对地面的平均压强为497千帕。2010年时倾斜角度为3.97度[17][18][19]，偏离地基外沿2.3米，顶层突出4.5米[20][21][6]。"})
    history_openai_format.append({"role": "user", "content": history[-1][0]})

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:11434/v1",
    )
    completion = client.chat.completions.create(
        model=model,
        messages=history_openai_format,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    start_time = time.time()

    for chunk in completion:
        history[-1][1] += chunk.choices[0].delta.content
        yield history, [[0, 0, 0, 0, 0]]

    end_time = time.time()
    response_time = end_time - start_time

    # Calculate token count and tokens per second
    completion_without_stream = client.chat.completions.create(
        model=model,
        messages=history_openai_format,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

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
    yield history, [
        [
            prompt_tokens_count,
            completion_tokens_count,
            total_token_count,
            response_time,
            total_token_count / response_time,
        ]
    ]


def chat_tab():
    chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False)

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=[],
        placeholder="Enter message or upload file...",
        show_label=False,
    )
    with gr.Row():
        # TODO: Query from installed models
        model_choice = gr.Dropdown(
            choices=["qwen:0.5b", "qwen:1.8b", "qwen:4b"],
            value="qwen:0.5b",
            label="Choose Model",
        )

        # Temperature slider
        temperature = gr.Slider(
            value=0.1,
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            label="Temperature",
        )

        # Max tokens slider
        max_tokens = gr.Slider(
            value=1024,
            minimum=32,
            maximum=4096,
            step=32,
            label="Max Tokens",
        )

    # Create a DataFrame (table) to display token and performance data
    table = gr.DataFrame(
        headers=[
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "response_time (s)",
            "tokens/s",
        ],
        datatype=["number"] * 5,
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(
        bot,
        [chatbot, model_choice, temperature, max_tokens],
        [chatbot, table],
        api_name="bot_response",
    )
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])


with gr.Blocks() as main_block:
    gr.Markdown("<h1><center>Build Your Own Chatbot with Local LLM Model</center></h1>")

    with gr.Tabs():
        with gr.Tab(label="Chat"):
            chat_tab()

        with gr.Tab(label="Manager"):
            manager_tab()

        # with gr.Tab(label="Monitor"):
        # monitor_tab()

        with gr.Tab(label="Files"):
            files_tab()

main_block.queue()
main_block.launch()
