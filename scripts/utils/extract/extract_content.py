from trafilatura import fetch_url, extract

downloaded = fetch_url('https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/')

# output main content and comments as plain text
result = extract(downloaded)
print(result)