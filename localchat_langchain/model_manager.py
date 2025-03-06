import requests
import pandas as pd
import gradio as gr

from utils.alert import show_info, show_warning

DEFAULT_PAGE_SIZE = 10
DEFAULT_PAGE_NUM = 1
DEFAULT_DOCUMENT_ID = 1


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
            show_warning(f"Expected a list, got: {type(models)}")
            return pd.DataFrame()

        data = {
            "Name": [model.get("name", "") for model in models],
            "Model": [model.get("model", "") for model in models],
            "Size (bytes)": [model.get("size", 0) for model in models],
            "Digest": [model.get("digest", "") for model in models],
            "Parameter Size": [
                model.get("details", {}).get("parameter_size", "") for model in models
            ],
            "Format": [model.get("details", {}).get("format", "") for model in models],
            "Quantization Level": [
                model.get("details", {}).get("quantization_level", "")
                for model in models
            ],
            "Modified At": [model.get("modified_at", "") for model in models],
        }

        df = pd.DataFrame(data)

        start = (page_number - 1) * page_size
        end = start + page_size
        show_info("Model list fetched successfully!")
        return df.iloc[start:end].reset_index(drop=True)

    except Exception as e:
        print(f"Error fetching model list: {e}")
        show_warning(f"Error fetching model list: {e}")
        return pd.DataFrame()


def model_manager_tab():
    gr.Markdown("## Model List")

    with gr.Row():
        # Left side - Document submission form
        with gr.Column(scale=1):
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

        # Right side - Document list display with delete support
        with gr.Column(scale=3):
            model_list = fetch_model_list(DEFAULT_PAGE_NUM, DEFAULT_PAGE_SIZE)
            model_list_table = gr.DataFrame(
                headers=[
                    "Name",
                    "Model",
                    "Modified At",
                    "Size (bytes)",
                    "Digest",
                    "Parameter Size",
                    "Quantization Level",
                ],
                datatype=["str", "str", "str", "number", "str", "str", "str"],
                label="Available Models",
                height=500,
                value=model_list,
            )

            fetch_models_button = gr.Button("Refresh Model")
            fetch_models_button.click(
                fn=fetch_model_list,
                inputs=[page_number_input, page_size_input],
                outputs=model_list_table,
            )


if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>LocalChat Model Manager</center></h1>")
        model_manager_tab()

    main_block.queue()
    main_block.launch()
