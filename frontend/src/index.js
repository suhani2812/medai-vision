import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/styles.css';
import App from './App';

// Create a root for your React application
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render your App component inside the root element
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
