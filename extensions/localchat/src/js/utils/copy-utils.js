/**
 * Code Copy Tool
 */

// Function to copy text to the clipboard
export function copyToClipboard(text, button) {
    // Process special escape characters for line breaks, carriage returns, and tabs
    const processedText = text.replace(/\\n/g, '\n')
                              .replace(/\\r/g, '\r')
                              .replace(/\\t/g, '\t');
    
    // Attempt to write the processed text to the clipboard
    navigator.clipboard.writeText(processedText).then(() => {
        // Save the original button text and update it to 'Copied'
        const originalText = button.textContent;
        button.textContent = "Copied!";
        
        // Restore the original button text after 2 seconds
        setTimeout(() => {
            button.textContent = originalText;
        }, 2000);
    }).catch(err => {
        // Log any errors that occur during the copy process
        console.error('Failed to copy: ', err);
        
        // Update button text to indicate an error occurred
        button.textContent = 'Error!';
        
        // Reset the button text back to 'Copy' after 2 seconds
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000);
    });
}

/**
 * Adds event listeners to enable the copy functionality
 */
export function addCopyListeners() {
    // Ensure the document body is loaded before adding listeners
    if (!document.body) {
        document.addEventListener('DOMContentLoaded', addCopyListeners);
        return;
    }
    
    // Use event delegation to listen for clicks on the document body
    document.body.addEventListener('click', (event) => {
        // Check if the clicked element is a copy button
        if (event.target.classList.contains('copy-button')) {
            const code = event.target.getAttribute('data-code');
            
            // If a valid code is found, process it
            if (code) {
                // Process special escape characters for line breaks, carriage returns, and tabs
                const processedCode = code.replace(/\\n/g, '\n')
                                         .replace(/\\r/g, '\r')
                                         .replace(/\\t/g, '\t');
                
                // Attempt to write the processed code to the clipboard
                navigator.clipboard.writeText(processedCode).then(() => {
                    // Save the original button text and update it to 'Copied'
                    const originalText = event.target.textContent;
                    event.target.textContent = "Copied!";
                    
                    // Restore the original button text after 2 seconds
                    setTimeout(() => {
                        event.target.textContent = originalText;
                    }, 2000);
                }).catch(err => {
                    // Log any errors that occur during the copy process
                    console.error('Failed to copy: ', err);
                });
            }
        }
    });
    
    console.log('Copy listeners have been successfully added.');
}
