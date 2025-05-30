// Listen for click events to enable code copying functionality
document.addEventListener('click', (event) => {
    // Check if the clicked element is a copy button
    if (event.target.classList.contains('copy-button')) {
        // Try to get the code from the 'data-code' attribute of the clicked button
        const code = event.target.getAttribute('data-code');
        
        if (code) {
            // Process special escape characters (e.g., newline, carriage return, tab)
            const processedCode = code.replace(/\\n/g, '\n')
                                     .replace(/\\r/g, '\r')
                                     .replace(/\\t/g, '\t');
            
            // Try to copy the processed code to the clipboard
            navigator.clipboard.writeText(processedCode).then(() => {
                const button = event.target;
                const originalText = button.textContent;
                
                // Update the button text to 'Copied!' after successful copy
                button.textContent = 'Copied!';
                
                // Revert the button text back to its original state after 2 seconds
                setTimeout(() => {
                    button.textContent = originalText;
                }, 2000);
            }).catch(err => {
                // Log any errors that occur during the copy operation
                console.error('Failed to copy: ', err);
            });
        } else {
            // If 'data-code' attribute is missing, look for a code block
            const codeBlock = event.target.closest('.code-block');
            if (codeBlock) {
                const codeElement = codeBlock.querySelector('code');
                if (codeElement) {
                    const code = codeElement.textContent;
                    
                    // Try to copy the code from the code block to the clipboard
                    navigator.clipboard.writeText(code).then(() => {
                        const button = event.target;
                        const originalText = button.textContent;
                        
                        // Update the button text to 'Copied!' after successful copy
                        button.textContent = 'Copied!';
                        
                        // Revert the button text back to its original state after 2 seconds
                        setTimeout(() => {
                            button.textContent = originalText;
                        }, 2000);
                    }).catch(err => {
                        // Log any errors that occur during the copy operation
                        console.error('Failed to copy: ', err);
                    });
                }
            }
        }
    }
});
