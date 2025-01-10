'use strict';

// With background scripts you can communicate with popup
// and contentScript files.
// For more information on background script,
// See https://developer.chrome.com/extensions/background_pages

// chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//   if (request.type === 'GREETINGS') {
//     const message = `Hi ${
//       sender.tab ? 'Con' : 'Pop'
//     }, my name is Bac. I am from Background. It's great to hear from you.`;

//     // Log message coming from the `request` parameter
//     console.log(request.payload.message);
//     // Send a response message
//     sendResponse({
//       message,
//     });
//   }
// });

// chrome.runtime.onCommand.addListener((command) => {
//   console.log("send_page_text command triggered", command);
//   if (command === "send_page_text") {
//     chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
//       const activeTab = tabs[0];
//       chrome.tabs.sendMessage(activeTab.id, { "message": "send_page_text" });
//     });
//   }
// })

// chrome.commands.onCommand.addListener(async (command) => {
//   console.log("send_page_text command triggered", command);
//   if (command === "send_page_text") {
//     const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
//     if (tab && tab.id) {
//       chrome.scripting.executeScript({
//         target: { tabId: tab.id },
//         files: ["content.js"]
//       });
//     }
//   }
// });

chrome.commands.onCommand.addListener(async (command) => {
  console.log('send_page_text command triggered');
  const tabs = await chrome.tabs.query({ currentWindow: true });
  // Sort tabs according to their index in the window.
  tabs.sort((a, b) => {
    return a.index < b.index;
  });
  const activeIndex = tabs.findIndex((tab) => {
    return tab.active;
  });
  const lastTab = tabs.length - 1;
  let newIndex = -1;
  if (command === 'send_page_text') {
    newIndex = activeIndex === lastTab ? 0 : activeIndex + 1;
  }
  // 'flip-tabs-backwards'
  else newIndex = activeIndex === 0 ? lastTab : activeIndex - 1;
  chrome.tabs.update(tabs[newIndex].id, { active: true, highlighted: true });
});