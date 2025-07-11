"""
CSS Styles for the AI 3D Asset Generator UI.

This module contains all the CSS styling for the Gradio interface.
"""

# Simplified, modern CSS
MODERN_CSS = """
/* Modern Variables */
:root {
    --primary-color: #2563eb;
    --primary-light: #3b82f6;
    --secondary-color: #64748b;
    --success-color: #059669;
    --warning-color: #d97706;
    --error-color: #dc2626;
    --background: #f8fafc;
    --surface: #ffffff;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    --radius: 8px;
    --radius-lg: 12px;
}

/* Base Styling */
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--background);
    color: var(--text-primary);
}

/* Header */
.app-header {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
    color: white;
    text-align: center;
    padding: 2rem;
    border-radius: var(--radius-lg);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
}

.app-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.app-subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    font-weight: 400;
}

/* Cards and Sections */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border-color);
}

/* Form Elements */
.form-input {
    border: 2px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    transition: all 0.2s ease !important;
}

.form-input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* Buttons */
.btn-primary {
    background: var(--primary-color) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    color: white !important;
    transition: all 0.2s ease !important;
}

.btn-primary:hover {
    background: var(--primary-light) !important;
    transform: translateY(-1px) !important;
}

.btn-secondary {
    background: transparent !important;
    border: 2px solid var(--secondary-color) !important;
    color: var(--secondary-color) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.btn-secondary:hover {
    background: var(--secondary-color) !important;
    color: white !important;
}

/* Status Messages */
.status-success {
    background: rgba(5, 150, 105, 0.1) !important;
    border: 1px solid var(--success-color) !important;
    color: var(--success-color) !important;
    padding: 0.75rem !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
    margin: 0.5rem 0 !important;
}

.status-error {
    background: rgba(220, 38, 38, 0.1) !important;
    border: 1px solid var(--error-color) !important;
    color: var(--error-color) !important;
    padding: 0.75rem !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
    margin: 0.5rem 0 !important;
    animation: shake 0.5s ease-in-out !important;
}

.status-error strong {
    font-weight: 600 !important;
    color: #b91c1c !important;
}

.status-error small {
    font-size: 0.85em !important;
    opacity: 0.8 !important;
    display: block !important;
    margin-top: 0.25rem !important;
}

.status-warning {
    background: rgba(217, 119, 6, 0.1) !important;
    border: 1px solid var(--warning-color) !important;
    color: var(--warning-color) !important;
    padding: 0.75rem !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
    margin: 0.5rem 0 !important;
}

/* Shake animation for errors */
@keyframes shake {
    0%, 20%, 50%, 80%, 100% {
        transform: translateX(0);
    }
    10%, 30%, 70%, 90% {
        transform: translateX(-3px);
    }
    40%, 60% {
        transform: translateX(3px);
    }
}

/* Progress Bar */
.progress-container {
    background: var(--surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1rem;
    margin: 1rem 0;
}

.progress-bar {
    background: var(--border-color);
    border-radius: 9999px;
    height: 8px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.progress-fill {
    background: var(--primary-color);
    height: 100%;
    border-radius: 9999px;
    transition: width 0.3s ease;
}

/* Model Viewer */
.model-viewer {
    border: 2px solid var(--border-color);
    border-radius: var(--radius-lg);
    min-height: 400px;
    background: var(--surface);
}

/* Results Tab Styles */
.asset-metadata {
    background: var(--surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1rem;
    margin: 0.5rem 0;
}

.asset-metadata h4 {
    margin: 0 0 1rem 0;
    color: var(--primary-color);
    font-size: 1.2rem;
}

.metadata-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.metadata-grid div {
    padding: 0.5rem;
    background: var(--background);
    border-radius: var(--radius-sm);
    font-size: 0.9rem;
}

/* Asset dropdown styling */
.dropdown-assets {
    margin-top: 1rem;
}

.dropdown-assets label {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.dropdown-assets select {
    border: 2px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem !important;
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}

.dropdown-assets select:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

.description {
    padding: 1rem;
    background: var(--background);
    border-radius: var(--radius);
    border-left: 4px solid var(--primary-color);
}

.no-asset {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
    font-style: italic;
}

/* Responsive Design */
@media (max-width: 768px) {
    .app-title {
        font-size: 2rem;
    }
    
    .section-card {
        padding: 1rem;
        margin-bottom: 1rem;
    }
}

/* Loading Animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--border-color);
    border-top: 3px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Gallery */
.gallery-container {
    background: var(--surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1rem;
}

/* Error Display */
.error-display {
    margin: 1rem 0 !important;
    border-radius: var(--radius) !important;
}

/* Form improvements */
.form-input {
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.form-input:focus-within {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* Generation button enhancements */
.btn-primary {
    position: relative !important;
    overflow: hidden !important;
}

.btn-primary:hover {
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
}

.btn-primary:disabled {
    opacity: 0.6 !important;
    cursor: not-allowed !important;
    transform: none !important;
}
"""
