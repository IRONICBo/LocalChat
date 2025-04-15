// Send message to Ollama via background script with streaming support
export async function sendMessageToOllama(message, history, onUpdate) {
    const settings = await getSettings();

    return new Promise((resolve, reject) => {
        let isResolved = false;
        let fullContent = '';

        chrome.runtime.sendMessage({
            action: 'sendMessageToOllama',
            message,
            history,
            systemPrompt: settings.systemPrompt
        }, (response) => {
            if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError);
                return;
            }

            if (response.error) {
                reject(new Error(response.error));
                return;
            }

            const messageId = response.messageId;

            const messageListener = (msg) => {
                if (msg.action === 'ollamaResponse' && msg.messageId === messageId) {
                    if (msg.done) {
                        if (!isResolved) {
                            isResolved = true;
                            resolve({ content: fullContent });
                            chrome.runtime.onMessage.removeListener(messageListener);
                        }
                    } else {
                        fullContent += msg.content;
                        if (onUpdate) {
                            onUpdate(msg.content, fullContent);
                        }
                    }
                } else if (msg.action === 'ollamaError' && msg.messageId === messageId) {
                    if (!isResolved) {
                        isResolved = true;
                        reject(new Error(msg.error));
                        chrome.runtime.onMessage.removeListener(messageListener);
                    }
                }
            };

            chrome.runtime.onMessage.addListener(messageListener);

            setTimeout(() => {
                if (!isResolved) {
                    isResolved = true;
                    reject(new Error('Request timed out'));
                    chrome.runtime.onMessage.removeListener(messageListener);
                }
            }, 30000);
        });
    });
}

// Get current settings
export async function getSettings() {
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({
            action: 'getSettings'
        }, (response) => {
            if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError);
            } else if (response.error) {
                reject(response.error);
            } else {
                resolve(response);
            }
        });
    });
}

// Update settings
export async function updateSettings(settings) {
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({
            action: 'updateSettings',
            settings: settings
        }, (response) => {
            if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError);
            } else if (response.error) {
                reject(response.error);
            } else {
                resolve(response);
            }
        });
    });
}