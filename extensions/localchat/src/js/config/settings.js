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

export async function updateSetting(key, value) {
    const settings = await getSettings();
    settings[key] = value;
    return updateSettings(settings);
}