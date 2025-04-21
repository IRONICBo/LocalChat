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

        if (role === 'user') {
            // contentElement.textContent = content;
            contentElement.innerHTML = renderMarkdown(content);
        } else {
            // render with markdown
            contentElement.innerHTML = renderMarkdown(content);

            if (typeof hljs !== 'undefined') {
                try {
                    contentElement.querySelectorAll('pre code').forEach((block) => {
                        try {
                            const originalContent = block.textContent;
                            block.textContent = originalContent;

                            hljs.highlightElement(block);
                        } catch (e) {
                            console.debug('Error highlighting individual code block:', e);
                        }
                    });
                } catch (e) {
                    console.debug('Error during code highlighting:', e);
                }
            }
        }

        messageElement.appendChild(contentElement);

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

            // regenerate button
            const regenerateButton = actionsElement.querySelector('.action-regenerate-button');
            regenerateButton.addEventListener('click', () => {
                regenerateResponse(messageElement);
            });

            // copy button
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

}