import docker
import pandas as pd
import gradio as gr
from datetime import datetime
import threading
import time

client = docker.from_env()
container = client.containers.get("ollama")

# Monitor CPU and memory usage of the container for ollama container
usage_data = pd.DataFrame(columns=["time", "cpu_usage", "mem_usage"])

# calculate CPU usage
def calculate_cpu_usage(stats):
    cpu_stats = stats['cpu_stats']
    precpu_stats = stats['precpu_stats']

    cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
    system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']

    if system_delta > 0:
        cpu_usage_percent = (cpu_delta / system_delta) * cpu_stats['online_cpus'] * 100
    else:
        cpu_usage_percent = 0.0

    return cpu_usage_percent

def update_usage_data():
    global usage_data
    while True:
        stats = container.stats(stream=False)
        cpu_usage = calculate_cpu_usage(stats)
        mem_usage = stats['memory_stats']['usage'] / 1024/1024/1024  # Convert to GB

        current_time = datetime.now()
        new_row = pd.DataFrame({"time": [current_time], "cpu_usage": [cpu_usage], "mem_usage": [mem_usage]})

        if not new_row.empty:
            usage_data = pd.concat([usage_data, new_row], ignore_index=True)
        if len(usage_data) > 200:
            usage_data = usage_data.iloc[-200:]

        time.sleep(0.5)

threading.Thread(target=update_usage_data, daemon=True).start()

def get_latest_usage_data():
    global usage_data
    print("running")
    if usage_data.empty:
        return pd.DataFrame(columns=["time", "cpu_usage"]), pd.DataFrame(columns=["time", "mem_usage"])
    else:
        cpu_usage_data = usage_data[["time", "cpu_usage"]]
        mem_usage_data = usage_data[["time", "mem_usage"]]
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
        height=300
    )

    # Create a line plot to show real-time memory usage
    mem_usage_plot = gr.LinePlot(
        value=usage_data[["time", "mem_usage"]],
        x="time",
        y="mem_usage",
        title="Real-time Memory Usage",
        label="Memory Usage Over Time",
        height=300
    )

    # Refresh the usage data every 0.5 seconds
    timer = gr.Timer(0.5)
    timer.tick(fn=get_latest_usage_data, outputs=[cpu_usage_plot, mem_usage_plot])


if __name__ == "__main__":
    app.launch()
