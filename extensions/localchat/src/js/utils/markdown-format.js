export function formatMarkdown(text) {
    if (!text) return '';

    let formatted = text;

    formatted = escapeHtml(formatted);

    // Replace ```code``` with a div containing a code block and a copy button
    formatted = formatted.replace(/```([\s\S]*?)```/g, (match, codeContent) => {
        const firstLine = codeContent.trim().split('\n')[0];
        let language = '';
        let code = codeContent;

        if (firstLine && !firstLine.includes(' ') && firstLine.length < 20) {
            language = firstLine;
            code = codeContent.substring(firstLine.length).trim();
        }

        const languageClass = language ? ` class="language-${language}"` : '';

        // Make sure newlines are escaped
        // Use textContent instead of innerHTML to avoid HTML parsing issues
        const escapedCode = code.replace(/&/g, '&amp;')
                               .replace(/</g, '&lt;')
                               .replace(/>/g, '&gt;')
                               .replace(/"/g, '&quot;')
                               .replace(/'/g, '&#39;');

        const dataCode = escapedCode.replace(/\n/g, '\\n')
                                   .replace(/\r/g, '\\r')
                                   .replace(/\t/g, '\\t');

        return `<div class="code-block">
            <div class="code-header">
                ${language ? `<span class="code-language">${language}</span>` : ''}
                <button class="copy-button" i18n-title="chat.copy" data-code="${dataCode}">Copy</button>
            </div>
            <pre><code${languageClass}>${escapedCode}</code></pre>
        </div>`;
    });

    formatted = formatted.replace(/`([^`]+)`/g, (match, code) => {
        return `<code>${code}</code>`;
    });

    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
}

function escapeHtml(text) {
    const htmlEntities = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    };

    const codeBlocks = [];
    text = text.replace(/```([\s\S]*?)```/g, (match) => {
        codeBlocks.push(match);
        return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
    });

    const inlineCodes = [];
    text = text.replace(/`([^`]+)`/g, (match) => {
        inlineCodes.push(match);
        return `__INLINE_CODE_${inlineCodes.length - 1}__`;
    });

    text = text.replace(/[&<>"']/g, (char) => htmlEntities[char]);

    codeBlocks.forEach((block, index) => {
        text = text.replace(`__CODE_BLOCK_${index}__`, block);
    });

    inlineCodes.forEach((code, index) => {
        text = text.replace(`__INLINE_CODE_${index}__`, code);
    });

    return text;
}