// Import the API service and markdown renderer
import { sendMessageToOllama, getSettings } from '../services/ollama-service.js';
import { sendMessageToOpenAI } from '../services/openai-service.js';
import { renderMarkdown } from '../utils/markdown-renderer.js';
import { t } from '../utils/i18n.js';

// Load AI Chat
export function loadAIChat(container) {
    // Chat history
    let chatHistory = [];

    // Current chat ID
    let currentChatId = null;

    let isGenerating = false;

    container.innerHTML = `
        <div class="chat-container">
            <div class="chat-header">
                <h2 data-i18n="chat.header">AI Chat</h2>
            </div>
            <div class="chat-messages" id="chat-messages">
                <!-- Welcome message will be added here -->
            </div>
            <div class="chat-input-wrapper">
                    <!--Disable now-->
                    <!--<div class="chat-actions">
                        <button id="new-chat-button" class="action-button" data-i18n-title="chat.newChat" title="New Chat">
                            <img src="assets/svg/new-chat.svg" alt="New Chat" class="button-icon">
                        </button>
                        <button id="history-button" class="action-button" data-i18n-title="chat.history" title="Chat History">
                            <img src="assets/svg/history.svg" alt="History" class="button-icon">
                        </button>
                    </div> -->
                <div class="chat-input-container">
                    <textarea id="chat-input" data-i18n-placeholder="chat.placeholder" placeholder="Type your message..." rows="1"></textarea>
                    <div class="chat-actions">
                        <button id="send-button" data-i18n-title="chat.send" title="Send">
                            <img src="assets/svg/send.svg" alt="Send" class="button-icon">
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <div id="history-popup" class="history-popup">
            <div class="history-popup-header">
                <h3 data-i18n="chat.historyTitle">Chat History</h3>
                <button id="close-history" class="icon-button">
                    <span>×</span>
                </button>
            </div>
            <div class="history-popup-content">
                <div id="history-list" class="history-list">
                    <!-- 历史记录将在这里显示 -->
                </div>
            </div>
        </div>
    `;

    // Get DOM elements
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const newChatButton = document.getElementById('new-chat-button');
    const historyButton = document.getElementById('history-button');
    const historyPopup = document.getElementById('history-popup');
    const closeHistoryButton = document.getElementById('close-history');
    const historyList = document.getElementById('history-list');

    // Add welcome message
    function addWelcomeMessage() {
        const welcomeElement = document.createElement('div');
        welcomeElement.className = 'message assistant welcome-message';

        const contentElement = document.createElement('div');
        contentElement.className = 'message-content';
        contentElement.innerHTML = renderMarkdown(t('chat.welcomeMessage', 'Hi, How can I help you today?'));

        welcomeElement.appendChild(contentElement);
        chatMessages.appendChild(welcomeElement);
    }

    // Function to add message to UI
    function addMessageToUI(role, content) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${role}`;

        const contentElement = document.createElement('div');
        contentElement.className = 'message-content';

        // 对于用户消息，使用textContent而不是innerHTML
        if (role === 'user') {
            contentElement.textContent = content; // 使用textContent确保HTML标签被显示为文本
        } else {
            // 对于助手消息，继续使用Markdown渲染
            contentElement.innerHTML = renderMarkdown(content);

            // 手动初始化代码高亮，使用 try-catch 捕获可能的错误
            if (typeof hljs !== 'undefined') {
                try {
                    contentElement.querySelectorAll('pre code').forEach((block) => {
                        try {
                            // 确保代码内容被正确转义
                            const originalContent = block.textContent;
                            block.textContent = originalContent;

                            hljs.highlightElement(block);
                        } catch (e) {
                            // 忽略单个代码块的高亮错误
                            console.debug('Error highlighting individual code block:', e);
                        }
                    });
                } catch (e) {
                    // 忽略整体高亮错误
                    console.debug('Error during code highlighting:', e);
                }
            }
        }

        messageElement.appendChild(contentElement);

        // 为助手消息添加重新生成按钮
        if (role === 'assistant') {
            const actionsElement = document.createElement('div');
            actionsElement.className = 'message-actions';
            actionsElement.innerHTML = `
                <button class="action-copy-button" title="${t('chat.copy')}">
                    <img src="assets/svg/copy.svg" alt="Copy" class="button-icon">
                </button>
                <button class="action-regenerate-button" title="${t('chat.regenerate')}">
                    <img src="assets/svg/refresh.svg" alt="Regenerate" class="button-icon">
                </button>
            `;
            messageElement.appendChild(actionsElement);

            // 添加重新生成功能
            const regenerateButton = actionsElement.querySelector('.action-regenerate-button');
            regenerateButton.addEventListener('click', () => {
                regenerateResponse(messageElement);
            });

            // 添加复制功能
            const copyButton = actionsElement.querySelector('.action-copy-button');
            copyButton.addEventListener('click', () => {
                copyToClipboard(contentElement.textContent);
            });
        }

        chatMessages.appendChild(messageElement);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return contentElement;
    }

    // Function to send message
    let streamingMessageElement = null;
    let codeBlocks = new Map(); // 用于跟踪代码块


    // 简化输入事件处理

    async function sendMessage() {
        const message = chatInput.value.trim();

        if (!message) return;

        // 如果 AI 正在生成回复，不允许发送新消息
        if (isGenerating) {
            console.log('AI is still generating a response, please wait');
            return;
        }

        // 删除欢迎消息（如果存在）
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        // Clear input
        chatInput.value = '';

        // Reset input height
        chatInput.style.height = 'auto';

        // Add user message to UI
        addMessageToUI('user', message);

        // Add user message to chat history
        chatHistory.push({
            role: 'user',
            content: message
        });

        // Create streaming message element
        const assistantMessageElement = document.createElement('div');
        assistantMessageElement.className = 'message assistant';

        const contentElement = document.createElement('div');
        contentElement.className = 'message-content';

        // Add typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';

        contentElement.appendChild(typingIndicator);
        assistantMessageElement.appendChild(contentElement);

        chatMessages.appendChild(assistantMessageElement);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Set streaming message element
        streamingMessageElement = contentElement;

        // Set generating state
        isGenerating = true;
        updateInputState();

        try {
            // Get settings
            const settings = await getSettings();

            // Choose API based on default AI provider
            let response;

            if (settings.defaultAI === 'openai') {
                // Import OpenAI service
                const { sendMessageToOpenAI, parseOpenAIStreamingResponse } = await import('../services/openai-service.js');

                // Use OpenAI API
                response = await sendMessageToOpenAI(message, chatHistory, settings.systemPrompt);

                // Handle streaming response
                if (response.streaming) {
                    let fullResponse = '';
                    let buffer = ''; // 用于存储可能被截断的数据

                    // 处理流式响应
                    while (true) {
                        try {
                            const { done, value } = await response.reader.read();

                            if (done) {
                                break;
                            }

                            // 解码响应
                            const chunk = response.decoder.decode(value, { stream: true });
                            buffer += chunk;

                            // 查找完整的数据行
                            const lines = buffer.split('\n');
                            buffer = lines.pop() || ''; // 保留最后一行（可能不完整）

                            for (const line of lines) {
                                if (line.trim().startsWith('data:')) {
                                    const content = parseOpenAIStreamingResponse(line);

                                    if (content && typeof content === 'string') {
                                        fullResponse += content;
                                        streamingMessageElement.innerHTML = renderMarkdown(fullResponse);

                                        // 应用代码高亮
                                        if (typeof hljs !== 'undefined') {
                                            try {
                                                streamingMessageElement.querySelectorAll('pre code').forEach((block) => {
                                                    try {
                                                        hljs.highlightElement(block);
                                                    } catch (e) {
                                                        console.debug('Error highlighting code block:', e);
                                                    }
                                                });
                                            } catch (e) {
                                                console.debug('Error during code highlighting:', e);
                                            }
                                        }

                                        // 自动滚动到底部
                                        scrollToBottom();
                                    }
                                }
                            }
                        } catch (error) {
                            console.error('Error reading stream:', error);
                            break;
                        }
                    }

                    // 处理buffer中可能剩余的数据
                    if (buffer.trim() && buffer.trim().startsWith('data:')) {
                        const content = parseOpenAIStreamingResponse(buffer);

                        if (content && typeof content === 'string') {
                            fullResponse += content;
                            streamingMessageElement.innerHTML = renderMarkdown(fullResponse);

                            // 应用代码高亮
                            if (typeof hljs !== 'undefined') {
                                try {
                                    streamingMessageElement.querySelectorAll('pre code').forEach((block) => {
                                        try {
                                            hljs.highlightElement(block);
                                        } catch (e) {
                                            console.debug('Error highlighting code block:', e);
                                        }
                                    });
                                } catch (e) {
                                    console.debug('Error during code highlighting:', e);
                                }
                            }

                            // 自动滚动到底部
                            scrollToBottom();
                        }
                    }

                    // 如果没有收到任何内容，显示错误消息
                    if (!fullResponse) {
                        console.error('No content received from OpenAI streaming response');
                        streamingMessageElement.innerHTML = '<div class="error-message">Error: No content received from OpenAI</div>';
                    } else {
                        // 将完整的响应添加到聊天历史
                        chatHistory.push({
                            role: 'assistant',
                            content: fullResponse
                        });

                        // 保存聊天历史
                        await saveCurrentChat();
                    }
                } else {
                    // Handle non-streaming response
                    // 直接更新 streamingMessageElement
                    if (streamingMessageElement) {
                        // 移除打字指示器
                        const typingIndicator = streamingMessageElement.querySelector('.typing-indicator');
                        if (typingIndicator) {
                            typingIndicator.remove();
                        }

                        // 更新内容
                        streamingMessageElement.innerHTML = renderMarkdown(response.fullResponse);

                        // 应用代码高亮
                        if (typeof hljs !== 'undefined') {
                            try {
                                streamingMessageElement.querySelectorAll('pre code').forEach((block) => {
                                    try {
                                        hljs.highlightElement(block);
                                    } catch (e) {
                                        console.debug('Error highlighting code block:', e);
                                    }
                                });
                            } catch (e) {
                                console.debug('Error during code highlighting:', e);
                            }
                        }

                        // 滚动到底部
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }

                    // Add to chat history
                    chatHistory.push({
                        role: 'assistant',
                        content: response.fullResponse
                    });


                }
                    // Save chat history
                await saveCurrentChat();

                // Reset generating state
                isGenerating = false;
                updateInputState();
            } else {
                // Default to Ollama API
                response = await sendMessageToOllama(message, chatHistory, (chunk, fullText) => {
                    // Remove typing indicator
                    const typingIndicator = streamingMessageElement.querySelector('.typing-indicator');
                    if (typingIndicator) {
                        typingIndicator.remove();
                    }

                    // Update content
                    streamingMessageElement.innerHTML = renderMarkdown(fullText);

                    // Apply code highlighting
                    if (typeof hljs !== 'undefined') {
                        try {
                            streamingMessageElement.querySelectorAll('pre code').forEach((block) => {
                                try {
                                    hljs.highlightElement(block);
                                } catch (e) {
                                    console.debug('Error highlighting code block:', e);
                                }
                            });
                        } catch (e) {
                            console.debug('Error during code highlighting:', e);
                        }
                    }

                    // Scroll to bottom
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                });

                // Add assistant message to chat history
                chatHistory.push({
                    role: 'assistant',
                    content: response.content
                });

                // Save chat history
                await saveCurrentChat();

                // Reset generating state
                isGenerating = false;
                updateInputState();
            }

            // Reset streaming message element
            streamingMessageElement = null;
        } catch (error) {
            console.error('Error sending message:', error);

            // Show error message
            if (streamingMessageElement) {
                streamingMessageElement.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
            }

            // Reset generating state
            isGenerating = false;
            updateInputState();

            // Reset streaming message element
            streamingMessageElement = null;
        }
    }
}