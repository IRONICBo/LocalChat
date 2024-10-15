import requests
import pandas as pd
import gradio as gr

def fetch_model_list(page_number, page_size):
    """
    Fetch model list from the API and return a paginated DataFrame.

    :param page_number: Page number for pagination
    :param page_size: Number of records per page
    :return: DataFrame containing the model list
    """
    api_url = "http://localhost:11434/api/tags"

    try:
        response = requests.get(api_url)
        response.raise_for_status()

        models = response.json()

        models = models.get("models", [])
        print(f"Model info: {models}")
        if not isinstance(models, list):
            print("Expected a list, got:", type(models))
            return pd.DataFrame()

        data = {
            "Name": [model.get("name", "") for model in models],
            "Model": [model.get("model", "") for model in models],
            "Size (bytes)": [model.get("size", 0) for model in models],
            "Digest": [model.get("digest", "") for model in models],
            "Parameter Size": [model.get("details", {}).get("parameter_size", "") for model in models],
            "Format": [model.get("details", {}).get("format", "") for model in models],
            "Quantization Level": [model.get("details", {}).get("quantization_level", "") for model in models],
            "Modified At": [model.get("modified_at", "") for model in models],
        }

        df = pd.DataFrame(data)

        # 实现分页
        start = (page_number - 1) * page_size
        end = start + page_size
        return df.iloc[start:end].reset_index(drop=True)

    except Exception as e:
        print(f"Error fetching model list: {e}")
        return pd.DataFrame()

with gr.Blocks() as app:
    gr.Markdown("## Model List")

    page_number_input = gr.Number(label="Page Number", value=1, precision=0)
    page_size_input = gr.Number(label="Page Size", value=10, precision=0)

    model_list_table = gr.DataFrame(
        headers=[
            "Name",
            "Model",
            "Modified At",
            "Size (bytes)",
            "Digest",
            "Parameter Size",
            "Quantization Level"
        ],
        datatype=["str", "str", "str", "number", "str", "str", "str"],
        label="Available Models",
        height=500,
    )

    fetch_models_button = gr.Button("Fetch Model List")
    fetch_models_button.click(
        fn=fetch_model_list,
        inputs=[page_number_input, page_size_input],
        outputs=model_list_table,
    )

if __name__ == "__main__":
    app.launch()
