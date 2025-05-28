let currentLanguage = 'en';

let translations = {};

let availableLanguages = [];

export async function initI18n(lang = 'en') {
    console.log('Initializing i18n...');

    try {
        const url = chrome.runtime.getURL('locale/languages.json');
        console.log(`Loading language configuration from: ${url}`);

        const response = await fetch(url);
        if (response.ok) {
            const data = await response.json();
            availableLanguages = data.languages || [];
            console.log('Available languages:', availableLanguages);
        }
    } catch (error) {
        console.error('Error loading language configuration:', error);
        availableLanguages = [
            { code: 'en', name: 'English' },
            { code: 'zh_cn', name: '简体中文' },
            { code: 'zh_tw', name: '繁體中文' }
        ];
    }

    try {
        const result = await new Promise((resolve) => {
            chrome.storage.local.get(['settings'], (result) => {
                resolve(result);
            });
        });

        console.log('Settings from storage:', result);

        if (result && result.settings && result.settings.language) {
            currentLanguage = result.settings.language;
            console.log(`Language from settings: ${currentLanguage}`);
        } else {
            const browserLang = navigator.language.toLowerCase();
            console.log(`Browser language: ${browserLang}`);

            const exactMatch = availableLanguages.find(lang => lang.code.toLowerCase() === browserLang);
            if (exactMatch) {
                currentLanguage = exactMatch.code;
            } else {
                const prefix = browserLang.split('-')[0];
                const prefixMatch = availableLanguages.find(lang =>
                    lang.code.toLowerCase().startsWith(prefix));
                if (prefixMatch) {
                    currentLanguage = prefixMatch.code;
                }
            }
            console.log(`Selected language: ${currentLanguage}`);
        }
    } catch (error) {
        console.error('Error loading language setting:', error);
    }

    await loadLanguage(currentLanguage);

    return translations;
}

export function getAvailableLanguages() {
    return availableLanguages;
}

export async function loadLanguage(lang) {
    try {
        const url = chrome.runtime.getURL(`locale/${lang}.json`);
        console.log(`Loading language file from: ${url}`);

        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load language file: ${response.status}`);
        }

        const data = await response.json();
        console.log('Language file loaded:', data);

        if (!data.app || !data.app.title) {
            console.error('Missing required translation key: app.title');
        }
        if (!data.app || !data.app.sidebar || !data.app.sidebar.chat) {
            console.error('Missing required translation key: app.sidebar.chat');
        }
        if (!data.app || !data.app.sidebar || !data.app.sidebar.settings) {
            console.error('Missing required translation key: app.sidebar.settings');
        }

        translations = data;
        currentLanguage = lang;

        chrome.storage.local.get(['settings'], (result) => {
            const settings = result.settings || {};
            settings.language = lang;
            chrome.storage.local.set({ settings });
        });

        updateDomTexts();

        document.dispatchEvent(new CustomEvent('languageChanged'));

        console.log(`Language changed to: ${lang}`);
        return translations;
    } catch (error) {
        console.error(`Error loading language file for ${lang}:`, error);
        if (lang !== 'en') {
            return loadLanguage('en');
        }
        return {};
    }
}

export function getCurrentLanguage() {
    return currentLanguage;
}

export function t(key, params = {}) {
    console.log(`Translating key: ${key}, current translations:`, translations);

    const keys = key.split('.');
    let value = translations;

    for (const k of keys) {
        if (value && typeof value === 'object' && k in value) {
            value = value[k];
        } else {
            console.warn(`Translation key not found: ${key}, current path: ${keys.slice(0, keys.indexOf(k)).join('.')}`);
            return key;
        }
    }

    if (typeof value === 'string') {
        return value.replace(/{([^}]+)}/g, (_, param) => {
            return params[param] !== undefined ? params[param] : `{${param}}`;
        });
    }

    console.warn(`Translation value is not a string: ${key}`);
    return key;
}

export function updateDomTexts() {
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        element.textContent = t(key);
    });

    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        element.placeholder = t(key);
    });

    document.querySelectorAll('[data-i18n-title]').forEach(element => {
        const key = element.getAttribute('data-i18n-title');
        element.title = t(key);
    });
}

document.addEventListener('languageChanged', () => {
    updateDomTexts();
});

function setupMutationObserver() {
    if (!document.body) {
        document.addEventListener('DOMContentLoaded', setupMutationObserver);
        return;
    }

    const observer = new MutationObserver((mutations) => {
        let needsUpdate = false;

        mutations.forEach(mutation => {
            if (mutation.type === 'childList' && mutation.addedNodes.length) {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1) { // Element node
                        if (node.hasAttribute('data-i18n') ||
                            node.hasAttribute('data-i18n-placeholder') ||
                            node.hasAttribute('data-i18n-title') ||
                            node.querySelector('[data-i18n], [data-i18n-placeholder], [data-i18n-title]')) {
                            needsUpdate = true;
                        }
                    }
                });
            }
        });

        if (needsUpdate) {
            updateDomTexts();
            console.log('DOM texts updated after mutation');
        }
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    console.log('MutationObserver setup complete');
}

setupMutationObserver();