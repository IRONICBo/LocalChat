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
