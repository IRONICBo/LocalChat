document.getElementById('summarizeBtn').addEventListener('click', function () {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      const currentTab = tabs[0];
      chrome.tabs.executeScript(currentTab.id, { code: "document.body.innerText" }, function (result) {
        const textToSummarize = result[0];
        chrome.runtime.sendMessage({ action: 'summarize', text: textToSummarize }, function (response) {
          const summary = response.summary || response.error;
          document.getElementById('summaryBubble').innerHTML = marked(summary); // Use markdown for formatting
        });
      });
    });
  });
