import { getSettings, updateSettings } from '../services/ollama-service.js';
import { t, getCurrentLanguage, loadLanguage, getAvailableLanguages } from '../utils/i18n.js';

// Load settings
export async function loadSettings(container) {
    const settings = await getSettings();

    // Get language options
    const languageOptions = await generateLanguageOptions();

    // Create settings UI with tabs
    container.innerHTML = `
        <div class="settings-container">
            <div class="settings-header">
                <h2 data-i18n="settings.header">Settings</h2>
            </div>

            <div class="settings-tabs">
                <button class="tab-button active" data-tab="general" data-i18n="settings.tabs.general">General</button>
                <button class="tab-button" data-tab="ollama" data-i18n="settings.tabs.ollama">Ollama</button>
                <button class="tab-button" data-tab="localchat">LocalChat</button>
                <button class="tab-button" data-tab="system-prompt" data-i18n="settings.tabs.systemPrompt">System Prompt</button>
            </div>

            <div class="settings-content">
                <!-- General Tab -->
                <div class="tab-content active" id="general-tab">
                    <div class="settings-section">
                        <h3 data-i18n="settings.sections.appearance.title">Appearance</h3>
                        <div class="settings-item">
                            <label data-i18n="settings.sections.appearance.theme.label">Theme</label>
                            <div class="settings-control">
                                <select id="theme-select">
                                    <option value="light" data-i18n="settings.sections.appearance.theme.light">Light</option>
                                    <option value="dark" data-i18n="settings.sections.appearance.theme.dark">Dark</option>
                                </select>
                            </div>
                        </div>
                        <div class="settings-item">
                            <label data-i18n="settings.sections.appearance.language.label">Language</label>
                            <div class="settings-control">
                                <select id="language-select">
                                    <option value="en">English</option>
                                    <option value="zh_cn">简体中文</option>
                                </select>
                            </div>
                        </div>
                        <div class="settings-item">
                            <h3 data-i18n="settings.sections.defaultAI.title">AI Settings</h3>
                            <div class="settings-item">
                                <label for="default-ai-select" data-i18n="settings.sections.defaultAI.label">Default AI Provider</label>
                                <div class="settings-control">
                                    <select id="default-ai-select">
                                        <option value="ollama" ${settings.defaultAI === 'ollama' ? 'selected' : ''} data-i18n="settings.sections.defaultAI.options.ollama">Ollama</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div class="settings-item checkbox-item">
                            <label>
                                <input type="checkbox" id="load-last-chat-checkbox">
                                <span data-i18n="settings.sections.appearance.loadLastChat">Load last chat on startup</span>
                            </label>
                        </div>
                    </div>
                </div>

                <!-- LocalChat Tab -->
                <div class="tab-content" id="localchat-tab">
                    <div class="settings-section">
                        <h3 data-i18n="settings.sections.localchat.title">LocalChat Settings</h3>
                        <div class="settings-item">
                            <label data-i18n="settings.sections.localchat.url.label">LocalChat URL</label>
                            <div class="url-input-container">
                                <div class="url-part">
                                    <label for="localchat-host" class="small-label" data-i18n="settings.sections.localchat.url.host">Host</label>
                                    <input type="text" id="localchat-host" placeholder="http://127.0.0.1">
                                </div>
                                <div class="url-part small">
                                    <label for="localchat-port" class="small-label" data-i18n="settings.sections.localchat.url.port">Port</label>
                                    <input type="text" id="localchat-port" placeholder="18080">
                                </div>
                                <div class="url-part">
                                    <label for="localchat-path" class="small-label" data-i18n="settings.sections.localchat.url.path">Path</label>
                                    <input type="text" id="localchat-path" placeholder="/api/generate">
                                </div>
                            </div>
                        </div>
                        <div class="settings-item">
                            <label class="checkbox-label">
                                <input type="checkbox" id="use-streaming" checked>
                                <span data-i18n="settings.sections.localchat.streaming">Enable auto capture</span>
                            </label>
                        </div>
                        <div class="settings-item">
                            <label class="checkbox-label">
                                <input type="checkbox" id="use-streaming" checked>
                                <span data-i18n="settings.sections.localchat.streaming">Enable streaming responses</span>
                            </label>
                        </div>
                    </div>
                    <div class="settings-section">
                        <h3 data-i18n="settings.sections.filtered.title">Filtered Settings</h3>
                        <div class="settings-item">
                            <label data-i18n="settings.sections.filtered.url.label">LocalChat URL</label>
                            <div class="url-input-container">
                                <div class="url-part">
                                    <label for="filtered-host" class="small-label" data-i18n="settings.sections.localchat.url.host">Host</label>
                                    <input type="text" id="filtered-host" placeholder="http://127.0.0.1">
                                </div>
                            </div>
                        </div>
                        <div id="connection-status" class="connection-status success">Please use , to split URLs</div>
                    </div>
                </div>

                <!-- Ollama Tab -->
                <div class="tab-content" id="ollama-tab">
                    <div class="settings-section">
                        <h3 data-i18n="settings.sections.ollama.title">Ollama Settings</h3>
                        <div class="settings-item">
                            <label data-i18n="settings.sections.ollama.url.label">Ollama URL</label>
                            <div class="url-input-container">
                                <div class="url-part">
                                    <label for="ollama-host" class="small-label" data-i18n="settings.sections.ollama.url.host">Host</label>
                                    <input type="text" id="ollama-host" placeholder="http://192.168.5.99">
                                </div>
                                <div class="url-part small">
                                    <label for="ollama-port" class="small-label" data-i18n="settings.sections.ollama.url.port">Port</label>
                                    <input type="text" id="ollama-port" placeholder="11434">
                                </div>
                                <div class="url-part">
                                    <label for="ollama-path" class="small-label" data-i18n="settings.sections.ollama.url.path">Path</label>
                                    <input type="text" id="ollama-path" placeholder="/api/generate">
                                </div>
                            </div>
                        </div>
                        <div class="settings-item">
                            <label for="ollama-model" data-i18n="settings.sections.ollama.model.label">Ollama Model</label>
                            <div class="model-select-container">
                                <select id="ollama-model" disabled>
                                    <option value="" data-i18n="settings.sections.ollama.model.loading">Loading models...</option>
                                </select>
                                <button id="refresh-models" class="icon-button" data-i18n-title="settings.buttons.refresh">
                                    <img src="assets/svg/refresh.svg" alt="Refresh" class="button-icon">
                                </button>
                            </div>
                        </div>
                        <div class="settings-item">
                            <label class="checkbox-label">
                                <input type="checkbox" id="use-proxy">
                                <span data-i18n="settings.sections.ollama.proxy">Use CORS proxy (try this if you get 403 errors)</span>
                            </label>
                        </div>
                        <div class="settings-item">
                            <label class="checkbox-label">
                                <input type="checkbox" id="use-streaming" checked>
                                <span data-i18n="settings.sections.ollama.streaming">Enable streaming responses</span>
                            </label>
                        </div>
                        <button id="test-connection" class="settings-button" data-i18n="settings.buttons.testConnection">Test Connection</button>
                        <button id="test-api" class="settings-button" data-i18n="settings.buttons.testApi">Test API</button>
                        <div id="connection-status" class="connection-status"></div>
                    </div>
                </div>

                <!-- OpenAI Tab -->
                <div class="tab-content" id="openai-tab" style="display: none;">
                    <div class="settings-section">
                        <h3 data-i18n="settings.sections.openai.title">OpenAI Settings</h3>

                        <div class="settings-item">
                            <label for="openai-api-key" data-i18n="settings.sections.openai.apiKey.label">API Key</label>
                            <div class="settings-control">
                                <input type="text" id="openai-api-key" placeholder="sk-..." value="${settings.openaiApiKey || ''}">
                            </div>
                        </div>

                        <div class="settings-item">
                            <label for="openai-base-url" data-i18n="settings.sections.openai.baseUrl.label">Base URL (Optional)</label>
                            <div class="settings-control">
                                <input type="text" id="openai-base-url" placeholder="https://api.openai.com/v1" value="${settings.openaiBaseUrl || ''}">
                            </div>
                        </div>

                        <div class="settings-item">
                            <label for="openai-model-select" data-i18n="settings.sections.openai.model.label">Model</label>
                            <div class="model-select-container">
                                <select id="openai-model-select">
                                    <option value="" data-i18n="settings.sections.openai.model.placeholder">Select a model</option>
                                    <option value="gpt-3.5-turbo" ${settings.openaiModel === 'gpt-3.5-turbo' ? 'selected' : ''}>GPT-3.5 Turbo</option>
                                    <option value="gpt-4" ${settings.openaiModel === 'gpt-4' ? 'selected' : ''}>GPT-4</option>
                                    <option value="gpt-4-turbo" ${settings.openaiModel === 'gpt-4-turbo' ? 'selected' : ''}>GPT-4 Turbo</option>
                                    <option value="custom" ${settings.openaiModel === 'custom' ? 'selected' : ''}>Custom</option>
                                </select>
                                <button id="refresh-openai-models" class="icon-button" data-i18n-title="settings.buttons.refresh">
                                    <img src="assets/svg/refresh.svg" alt="Refresh" class="button-icon">
                                </button>
                            </div>
                        </div>

                        <div class="settings-item">
                            <label for="openai-custom-model" data-i18n="settings.sections.openai.customModel.label">Custom Model</label>
                            <div class="settings-control">
                                <input type="text" id="openai-custom-model" placeholder="custom-model-name" value="${settings.openaiCustomModel || ''}">
                            </div>
                        </div>

                        <div class="settings-item">
                            <button id="test-openai-connection" class="settings-button" data-i18n="settings.buttons.testConnection">Test Connection</button>
                            <div id="openai-connection-status" class="settings-status"></div>
                        </div>
                    </div>
                </div>

                <!-- System Prompt Tab -->
                <div class="tab-content" id="system-prompt-tab">
                    <div class="settings-section">
                        <h3 data-i18n="settings.sections.systemPrompt.title">System Prompt</h3>
                        <div class="settings-item">
                            <label for="system-prompt" data-i18n="settings.sections.systemPrompt.label">System Prompt</label>
                            <textarea id="system-prompt" rows="8" placeholder="Enter system prompt here..." data-i18n-placeholder="settings.sections.systemPrompt.placeholder"></textarea>
                            <p class="settings-help" data-i18n="settings.sections.systemPrompt.help">
                                The system prompt is sent to the AI at the beginning of each conversation to set the context and behavior.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="settings-actions">
            <button id="reset-settings" class="settings-button secondary" data-i18n-title="settings.buttons.reset">
                <img src="assets/svg/reset.svg" alt="Reset" class="button-icon">
            </button>
            <button id="save-settings" class="settings-button primary" data-i18n="settings.buttons.save">Save Settings</button>
        </div>
    `;
}

async function generateLanguageOptions() {
    const languages = getAvailableLanguages();
    const currentLang = getCurrentLanguage();

    let options = '';
    languages.forEach(lang => {
        const selected = lang.code === currentLang ? 'selected' : '';
        options += `<option value="${lang.code}" ${selected}>${lang.name}</option>`;
    });

    return options;
}