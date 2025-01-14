from trafilatura import fetch_url, extract

downloaded = fetch_url('https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/')

# output main content and comments as plain text
result = extract(downloaded)
print(result)

markdown = extract(downloaded, output_format="markdown", with_metadata=True)
print(markdown)

# Convert local html file to markdown
with open("demo.html", "r") as f:
    html = f.read()
    markdown = extract(html, output_format="markdown", with_metadata=True)
    with open("demo.md", "w") as f:
        f.write(markdown)

with open("test_blog.html", "r") as f:
    html = f.read()
    markdown = extract(html, output_format="markdown", with_metadata=True)
    with open("test_blog.md", "w") as f:
        f.write(markdown)

with open("test_wikipedia.html", "r") as f:
    html = f.read()
    markdown = extract(html, output_format="markdown", with_metadata=True)
    with open("test_wikipedia.md", "w") as f:
        f.write(markdown)