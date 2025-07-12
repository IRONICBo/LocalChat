// Import the API service and markdown renderer
import { sendMessageToOllama, getSettings } from '../services/ollama-service.js';
import { sendMessageToOpenAI } from '../services/openai-service.js';
import { renderMarkdown } from '../utils/markdown-renderer.js';
import { t } from '../utils/i18n.js';

// Load AI Chat
export function loadAIChat(container) {
}