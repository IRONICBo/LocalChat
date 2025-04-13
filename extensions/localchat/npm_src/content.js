// 提取页面的纯文本内容
function getPageText() {
    // const bodyText = document.body.innerText || "";
    // return bodyText.trim();
    return document.body.innerHTML.trim();
  }

  // 发送 POST 请求到 API
  async function sendToAPI(content) {
    const apiURL = "http://127.0.0.1:8080/upload"; // 替换为你的 upload API 地址

    try {
      const response = await fetch(apiURL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          content: content,
          url: window.location.href,
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log("Response from API:", result);
        alert("内容已成功发送到服务器！");
      } else {
        console.error("Failed to send data:", response.statusText);
        alert("发送失败，请检查 API 是否可用！");
      }
    } catch (error) {
      console.error("Error while sending data to API:", error);
      alert("发送过程中发生错误，请查看控制台日志！");
    }
  }

  // 捕获页面文本并发送到 API
  const pageText = getPageText();
  sendToAPI(pageText);
