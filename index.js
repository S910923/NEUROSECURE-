import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Hide the HTML loading screen once React has painted
if (typeof window.__nsHideLoader === 'function') {
  window.__nsHideLoader();
}
