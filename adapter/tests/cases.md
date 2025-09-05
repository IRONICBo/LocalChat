
# case 1
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "请帮我草拟一封邮件，给我的同事John Doe（john.doe@example.com）说明今天会议的主要内容，会议中提到了我们在123 Main St, New York, NY 10001的办公室的扩展计划。"
      }
    ]

  }'


# case 2
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "我收到了一封来自客户Jane Smith（jane.smith@example.com）的邮件，请帮我自动回复。她提到了我们在伦敦的办公室（456 Queen St, London, UK SW1A 1AA）的项目进展。"
      }
    ]

  }'


# case 3
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "请帮我润色这封邮件，让语气更正式。邮件中提到了我们的客户Michael Johnson（michael.j@example.com）和他在多伦多的地址（789 King St, Toronto, ON M5H 2N2）。"
      }
    ]

  }'


# case 4
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "请帮我总结这封长邮件的重点。邮件中包含了我们在旧金山办公室（101 Market St, San Francisco, CA 94105）的季度报告。"
      }
    ]

  }'


# case 5
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "我在波士顿（Boston, MA 02108），今天天气适合推荐穿什么衣服？"
      }
    ]

  }'


# case 6
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "请帮我填写这个实验休假申请表。我的姓名是Emily Brown，身份证号是123456789，亲戚关系是我的兄弟James Brown。"
      }
    ]

  }'


# case 7
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "今天在123 BBQ St, Los Angeles, CA 90001吃烧烤拉肚子，我需要如何缓解？"
      }
    ]

  }'


# case 8
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "帮我给实验室群组的张同学发送消息，告知他我因生病请假。我的联系方式是123-456-7890。"
      }
    ]

  }'


# case 9
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Please send a message to Zhang in the lab group informing him that I am taking sick leave. My contact number is 1234567890."
      }
    ]

  }'

