import docker
import pandas as pd
import gradio as gr
from datetime import datetime
import threading
import time

from models import SessionLocal, ChatbotUsage
from op import get_usage_by_model_paginated

client = docker.from_env()
container = client.containers.get("ollama")

# Monitor CPU and memory usage of the container for ollama container
usage_data = pd.DataFrame(columns=["time", "cpu_usage", "mem_usage"])


def fetch_token_usage(model_name, page_number, page_size):
    """
    Fetch token usage history from the database for a specific model

    :param model_name: model name
    :param page_number: starting page number
    :param page_size: number of records per page
    :return: DataFrame containing token usage history
    """
    db = SessionLocal()
    try:
        records = get_usage_by_model_paginated(
            db, model_name, int(page_number), int(page_size)
        )
        # Convert the list of records into a dictionary for DataFrame creation
        data = {
            "Model": [record.model for record in records],
            "Temperature": [record.temperature for record in records],
            "Max Tokens": [record.max_tokens for record in records],
            "Total Tokens": [record.total_token_count for record in records],
            "Completion Tokens": [record.completion_tokens_count for record in records],
            "Prompt Tokens": [record.prompt_tokens_count for record in records],
            "Response Time (s)": [record.response_time for record in records],
        }
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Failed to fetch token usage history: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if an error occurs
    finally:
        db.close()


# calculate CPU usage
def calculate_cpu_usage(stats):
    cpu_stats = stats["cpu_stats"]
    precpu_stats = stats["precpu_stats"]

    cpu_delta = (
        cpu_stats["cpu_usage"]["total_usage"] - precpu_stats["cpu_usage"]["total_usage"]
    )
    system_delta = cpu_stats["system_cpu_usage"] - precpu_stats["system_cpu_usage"]

    if system_delta > 0:
        cpu_usage_percent = (cpu_delta / system_delta) * cpu_stats["online_cpus"] * 100
    else:
        cpu_usage_percent = 0.0

    return cpu_usage_percent


def update_usage_data():
    global usage_data
    while True:
        stats = container.stats(stream=False)
        cpu_usage = calculate_cpu_usage(stats)
        mem_usage = stats["memory_stats"]["usage"] / 1024 / 1024 / 1024  # Convert to GB

        current_time = datetime.now()
        new_row = pd.DataFrame(
            {"time": [current_time], "cpu_usage": [cpu_usage], "mem_usage": [mem_usage]}
        )

        if not new_row.empty and not new_row.isna().all().all():
            usage_data = pd.concat([usage_data, new_row], ignore_index=True)
        if len(usage_data) > 200:
            usage_data = usage_data.iloc[-200:]

        time.sleep(1)


threading.Thread(target=update_usage_data, daemon=True).start()


def get_latest_usage_data():
    global usage_data
    print("running")
    if usage_data.empty:
        return pd.DataFrame(columns=["time", "cpu_usage"]), pd.DataFrame(
            columns=["time", "mem_usage"]
        )
    else:
        cpu_usage_data = usage_data[["time", "cpu_usage"]]
        mem_usage_data = usage_data[["time", "mem_usage"]]
        print("cpu_usage_data", cpu_usage_data)
        print("mem_usage_data", mem_usage_data)
        return cpu_usage_data, mem_usage_data


with gr.Blocks() as app:
    gr.Markdown("## Real-time LLM Serving System Monitoring")

    # Create a line plot to show real-time CPU usage
    cpu_usage_plot = gr.LinePlot(
        value=usage_data[["time", "cpu_usage"]],
        x="time",
        y="cpu_usage",
        title="Real-time CPU Usage",
        label="CPU Usage Over Time",
        x_title="Time/s",
        y_title="CPU/%",
        height=300,
    )

    # Create a line plot to show real-time memory usage
    mem_usage_plot = gr.LinePlot(
        value=usage_data[["time", "mem_usage"]],
        x="time",
        y="mem_usage",
        title="Real-time Memory Usage",
        label="Memory Usage Over Time",
        x_title="Time/s",
        y_title="Memory/GB",
        height=300,
    )

    gr.Markdown("## Chatbot Token Metric History")

    # History of model token usage
    with gr.Row():
        # TODO: Support multiple models
        model_name_input = gr.Dropdown(
            choices=["qwen:0.5b"],
            label="Select Model Name",
            value="qwen:0.5b",
        )
        page_number_input = gr.Number(label="Page Number", value=1, precision=0)
        page_size_input = gr.Number(label="Page Size", value=10, precision=0)

    token_usage_table = gr.DataFrame(
        headers=[
            "Model",
            "Temperature",
            "Max Tokens",
            "Total Tokens",
            "Completion Tokens",
            "Prompt Tokens",
            "Response Time (s)",
        ],
        datatype=["str", "number", "number", "number", "number", "number", "number"],
        label="Token Usage History",
    )

    # fetch token usage history button
    query_button = gr.Button("Fetch Token Usage History")
    query_button.click(
        fn=fetch_token_usage,
        inputs=[model_name_input, page_number_input, page_size_input],
        outputs=token_usage_table,
    )

    # Refresh the usage data every 0.5 seconds
    timer = gr.Timer(1)
    timer.tick(fn=get_latest_usage_data, outputs=[cpu_usage_plot, mem_usage_plot])


if __name__ == "__main__":
    app.queue()
    app.launch()
