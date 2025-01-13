import io
import os
import shutil
from markitdown import MarkItDown


# This demo is only in python 3.10+
markitdown = MarkItDown()

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), ".")
WIKIPEDIA_TEST_URL = "https://en.wikipedia.org/wiki/Microsoft"
BLOG_TEST_URL = "https://microsoft.github.io/autogen/blog/2023/04/21/LLM-tuning-math"

result = markitdown.convert(
    os.path.join(TEST_FILES_DIR, "test_wikipedia.html"),
    url=WIKIPEDIA_TEST_URL
)
print(result.text_content)
with open(os.path.join(TEST_FILES_DIR, "test_wikipedia.md"), "w") as f:
    f.write(result.text_content)

result = markitdown.convert(
    os.path.join(TEST_FILES_DIR, "test_blog.html"),
    url=BLOG_TEST_URL
)
print(result.text_content)
with open(os.path.join(TEST_FILES_DIR, "test_blog.md"), "w") as f:
    f.write(result.text_content)

result = markitdown.convert(
    os.path.join(TEST_FILES_DIR, "demo.html"),
    url=BLOG_TEST_URL
)
with open(os.path.join(TEST_FILES_DIR, "demo.md"), "w") as f:
    f.write(result.text_content)