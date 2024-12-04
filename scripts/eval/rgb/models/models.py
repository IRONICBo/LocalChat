import openai


class OllamaModel:
    def __init__(
        self, api_key="EMPTY", url="http://localhost:11434/v1", model="qwen:0.5b"
    ):
        self.url = url
        self.model = model
        self.API_KEY = api_key
        self.client = openai.Client(api_key=api_key, base_url=url)

    def generate(
        self,
        text: str,
        temperature=0.7,
        system="You are a helpful assistant. You can help me by answering my questions. You can also ask me questions.",
        top_p=1,
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            top_p=top_p,
        )
        answer = response.choices[0].message.content.strip()
        return answer
