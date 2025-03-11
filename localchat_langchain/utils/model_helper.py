import requests


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
