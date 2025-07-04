window.addEventListener('error', function(event) {
    if (event.filename && event.filename.includes('highlight')) {
        event.preventDefault();
        return false;
    }
});

window.addEventListener('unhandledrejection', function(event) {
    if (event.reason && event.reason.stack &&
        event.reason.stack.includes('highlight')) {
        event.preventDefault();
        return false;
    }
});