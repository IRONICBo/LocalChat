from litellm import completion
import os

# For gemini
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"
response = completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}],
)
print(response)
