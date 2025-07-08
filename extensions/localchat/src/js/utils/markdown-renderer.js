import { t } from './i18n.js';

// Suppress highlight.js warnings
(function suppressHighlightJsWarnings() {
    const originalWarn = console.warn;
    const originalError = console.error;
    const originalLog = console.log;

    function isHighlightJsWarning(args) {
        if (args.length === 0) return false;

        if (typeof args[0] === 'string') {
            return args[0].includes('unescaped HTML') ||
                   args[0].includes('highlight.js/wiki/security');
        }

        if (args[0] instanceof HTMLElement ||
            (args[0] && args[0].toString && args[0].toString() === '[object HTMLElement]')) {
            return true;
        }

        return false;
    }

    console.warn = function(...args) {
        if (!isHighlightJsWarning(args)) {
            originalWarn.apply(console, args);
        }
    };

    console.error = function(...args) {
        if (args.length > 0 && typeof args[0] === 'string' &&
            (args[0].includes('highlight.js') || args[0].includes('Highlighting'))) {
            return;
        }
        originalError.apply(console, args);
    };

    console.log = function(...args) {
        if (args.length > 0 && typeof args[0] === 'string' &&
            (args[0].includes('Highlighting code with language') ||
             args[0].includes('highlight.js loaded') ||
             args[0].includes('Highlight successful') ||
             args[0].includes('Using auto highlight'))) {
            return;
        }
        originalLog.apply(console, args);
    };
})();

let hljs;
let hljsLoaded = false;
let hljsLanguages = [];

// Import marked and DOMPurify
function initHighlight() {
    if (typeof window.hljs !== 'undefined') {
        hljs = window.hljs;
        hljsLoaded = true;
        hljsLanguages = hljs.listLanguages();
        console.log('highlight.js loaded, available languages:', hljsLanguages);
    } else {
        console.warn('highlight.js not loaded');
    }
}

initHighlight();

if (!hljsLoaded) {
    window.addEventListener('DOMContentLoaded', initHighlight);
}

// Import marked and DOMPurify
marked.setOptions({
    highlight: function(code, lang) {
        console.log(`Highlighting code with language: ${lang}`);
        if (lang && hljs.getLanguage(lang)) {
            try {
                const result = hljs.highlight(code, {
                    language: lang,
                    ignoreIllegals: true
                }).value;
                console.log('Highlight successful');
                return result;
            } catch (e) {
                console.error('Highlight error:', e);
            }
        }
        console.log('Using auto highlight');
        return hljs.highlightAuto(code).value;
    },
    langPrefix: 'hljs language-',
    breaks: true,
    gfm: true
});

// Import DOMPurify
export function renderMarkdown(markdown) {
    if (!markdown) return '';

    try {
        let processedMarkdown = markdown;

        const htmlCodeBlockRegex = /```(html|xml)([\s\S]*?)```/g;
        const htmlCodeBlocks = [];

        processedMarkdown = processedMarkdown.replace(htmlCodeBlockRegex, (match, lang, code) => {
            const placeholder = `HTML_CODE_BLOCK_${htmlCodeBlocks.length}`;
            htmlCodeBlocks.push({ lang, code: code.trim() });
            return '```' + lang + '\n' + placeholder + '\n```';
        });

        processedMarkdown = preprocessMarkdown(processedMarkdown);

        let html = marked.parse(processedMarkdown);

        htmlCodeBlocks.forEach((block, index) => {
            const placeholder = `HTML_CODE_BLOCK_${index}`;
            const escapedCode = escapeHtml(block.code);
            html = html.replace(placeholder, escapedCode);
        });

        const sanitized = DOMPurify.sanitize(html);

        const processed = processCodeBlocks(sanitized);

        return processed;
    } catch (error) {
        console.error('Error rendering markdown:', error);
        return `<pre>${escapeHtml(markdown)}</pre>`;
    }
}

// Preprocess markdown to escape HTML in code blocks
function preprocessMarkdown(markdown) {
    const codeBlockRegex = /```([\s\S]*?)```/g;

    return markdown.replace(codeBlockRegex, (match, codeContent) => {
        if (match.startsWith('```html') || match.startsWith('```xml')) {
            return match;
        } else {
            return '```' + codeContent.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '```';
        }
    });
}

// Process code blocks in the HTML
function processCodeBlocks(html) {
    try {
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;

        tempDiv.querySelectorAll('pre code[class^="language-"]').forEach(codeElement => {
            try {
                const pre = codeElement.parentElement;
                const codeBlock = document.createElement('div');
                codeBlock.className = 'code-block';

                const langClass = Array.from(codeElement.classList).find(cls => cls.startsWith('language-'));
                const language = langClass ? langClass.replace('language-', '') : '';

                const header = document.createElement('div');
                header.className = 'code-header';

                const langLabel = document.createElement('span');
                langLabel.className = 'code-language';
                langLabel.textContent = language || 'text';
                header.appendChild(langLabel);

                const copyButton = document.createElement('button');
                copyButton.className = 'copy-button';
                copyButton.textContent = t('chat.copy');

                const codeContent = codeElement.textContent;
                copyButton.setAttribute('data-code', codeContent);
                header.appendChild(copyButton);

                pre.parentNode.insertBefore(codeBlock, pre);
                codeBlock.appendChild(header);
                codeBlock.appendChild(pre);

                if (hljsLoaded) {
                    if (!codeElement.id) {
                        const codeId = `code-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                        codeElement.setAttribute('id', codeId);
                    }

                    if (!codeElement.classList.contains('hljs')) {
                        try {
                            const originalContent = codeElement.textContent;
                            codeElement.textContent = originalContent;

                            if (language === '' || hljsLanguages.includes(language)) {
                                hljs.highlightElement(codeElement);
                            }
                        } catch (error) {
                            console.error('Error highlighting code:', error);
                        }
                    }
                }
            } catch (error) {
                console.error('Error processing code block:', error);
            }
        });

        tempDiv.querySelectorAll('pre code:not([class^="language-"])').forEach(codeElement => {
            const pre = codeElement.parentElement;
            const codeBlock = document.createElement('div');
            codeBlock.className = 'code-block';

            const header = document.createElement('div');
            header.className = 'code-header';

            const langLabel = document.createElement('span');
            langLabel.className = 'code-language';
            langLabel.textContent = 'text';
            header.appendChild(langLabel);

            const copyButton = document.createElement('button');
            copyButton.className = 'copy-button';
            copyButton.textContent = t('chat.copy');

            const codeContent = codeElement.textContent;
            copyButton.setAttribute('data-code', codeContent);

            header.appendChild(copyButton);

            pre.parentNode.insertBefore(codeBlock, pre);
            codeBlock.appendChild(header);
            codeBlock.appendChild(pre);

            if (hljsLoaded && !codeElement.classList.contains('hljs')) {
                const codeId = `code-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                codeElement.setAttribute('id', codeId);

                const originalContent = codeElement.textContent;
                codeElement.textContent = originalContent;

                hljs.highlightElement(codeElement);
            }
        });

        return tempDiv.innerHTML;
    } catch (error) {
        console.error('Error processing code blocks:', error);
        return html;
    }
}

// Apply code highlighting to elements
export function applyCodeHighlight(element) {
    if (!hljsLoaded) return;

    element.querySelectorAll('pre code:not(.hljs)').forEach(block => {
        hljs.highlightElement(block);
    });
}

function escapeHtml(text) {
    const htmlEntities = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    };

    return text.replace(/[&<>"']/g, (char) => htmlEntities[char]);
}