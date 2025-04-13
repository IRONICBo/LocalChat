document.addEventListener('DOMContentLoaded', function () {
    // Load saved settings
    chrome.storage.sync.get(['enablePlugin', 'urlFilter', 'blockCurrentPage', 'apiUrl'], function (data) {
      document.getElementById('enablePlugin').checked = data.enablePlugin || false;
      document.getElementById('urlFilter').value = data.urlFilter || '';
      document.getElementById('blockCurrentPage').checked = data.blockCurrentPage || false;
      document.getElementById('apiUrl').value = data.apiUrl || '';
    });

    // Save settings when the button is clicked
    document.getElementById('saveSettings').addEventListener('click', function () {
      const enablePlugin = document.getElementById('enablePlugin').checked;
      const urlFilter = document.getElementById('urlFilter').value;
      const blockCurrentPage = document.getElementById('blockCurrentPage').checked;
      const apiUrl = document.getElementById('apiUrl').value;

      chrome.storage.sync.set({
        enablePlugin,
        urlFilter,
        blockCurrentPage,
        apiUrl
      }, function () {
        alert('Settings saved!');
      });
    });
  });
