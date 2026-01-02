/**
 * Face Blur - Frontend Application
 * Handles video upload, processing status, and download
 */

// Configuration
const CONFIG = {
    // API URL - Update this with your Oracle Cloud VM IP or domain
    API_URL: 'http://localhost:8000',
    
    // File constraints
    MAX_FILE_SIZE: 500 * 1024 * 1024, // 500MB
    MAX_DURATION_SECONDS: 300, // 5 minutes
    ALLOWED_EXTENSIONS: ['.mp4', '.mov', '.avi'],
    ALLOWED_MIME_TYPES: ['video/mp4', 'video/quicktime', 'video/x-msvideo'],
    
    // Polling settings
    STATUS_POLL_INTERVAL: 1000, // 1 second
    MAX_POLL_ATTEMPTS: 3600, // 60 minutes max (for accurate face detection)
    
    // Retry settings
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000, // 1 second
};

// State
const state = {
    selectedFile: null,
    jobId: null,
    pollInterval: null,
    pollAttempts: 0,
    isUploading: false,
    isProcessing: false,
};

// DOM Elements
const elements = {
    uploadSection: document.getElementById('upload-section'),
    progressSection: document.getElementById('progress-section'),
    completeSection: document.getElementById('complete-section'),
    errorSection: document.getElementById('error-section'),
    uploadZone: document.getElementById('upload-zone'),
    fileInput: document.getElementById('file-input'),
    filePreview: document.getElementById('file-preview'),
    fileName: document.getElementById('file-name'),
    fileSize: document.getElementById('file-size'),
    btnRemove: document.getElementById('btn-remove'),
    btnUpload: document.getElementById('btn-upload'),
    btnCancel: document.getElementById('btn-cancel'),
    btnDownload: document.getElementById('btn-download'),
    btnNew: document.getElementById('btn-new'),
    btnRetry: document.getElementById('btn-retry'),
    progressBar: document.getElementById('progress-bar'),
    progressFill: document.getElementById('progress-fill'),
    progressPercentage: document.getElementById('progress-percentage'),
    progressStatus: document.getElementById('progress-status'),
    progressDetail: document.getElementById('progress-detail'),
    errorMessage: document.getElementById('error-message'),
    loadingOverlay: document.getElementById('loading-overlay'),
};

// Initialize application
function init() {
    // Set API URL from environment or use default
    if (typeof VITE_API_URL !== 'undefined') {
        CONFIG.API_URL = VITE_API_URL;
    }
    
    // Try to get API URL from meta tag (for static deployment)
    const apiUrlMeta = document.querySelector('meta[name="api-url"]');
    if (apiUrlMeta) {
        CONFIG.API_URL = apiUrlMeta.content;
    }
    
    setupEventListeners();
    console.log('Face Blur App initialized');
    console.log('API URL:', CONFIG.API_URL);
}

// Setup event listeners
function setupEventListeners() {
    // File input change
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Upload zone click
    elements.uploadZone.addEventListener('click', () => elements.fileInput.click());
    
    // Drag and drop
    elements.uploadZone.addEventListener('dragover', handleDragOver);
    elements.uploadZone.addEventListener('dragleave', handleDragLeave);
    elements.uploadZone.addEventListener('drop', handleDrop);
    
    // Buttons
    elements.btnRemove.addEventListener('click', handleRemoveFile);
    elements.btnUpload.addEventListener('click', handleUpload);
    elements.btnCancel.addEventListener('click', handleCancel);
    elements.btnDownload.addEventListener('click', handleDownload);
    elements.btnNew.addEventListener('click', handleNewVideo);
    elements.btnRetry.addEventListener('click', handleRetry);
    
    // Prevent default drag behavior on document
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

// File handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        validateAndSetFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadZone.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadZone.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadZone.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file) {
        validateAndSetFile(file);
    }
}

function validateAndSetFile(file) {
    // Check file extension
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    if (!CONFIG.ALLOWED_EXTENSIONS.includes(extension)) {
        showError(`Invalid file type. Please upload ${CONFIG.ALLOWED_EXTENSIONS.join(', ')} files.`);
        return;
    }
    
    // Check file size
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showError(`File too large. Maximum size is ${formatFileSize(CONFIG.MAX_FILE_SIZE)}.`);
        return;
    }
    
    // Set the file
    state.selectedFile = file;
    showFilePreview(file);
}

function showFilePreview(file) {
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatFileSize(file.size);
    elements.filePreview.classList.remove('hidden');
    elements.btnUpload.classList.remove('hidden');
}

function handleRemoveFile(event) {
    event.stopPropagation();
    state.selectedFile = null;
    elements.fileInput.value = '';
    elements.filePreview.classList.add('hidden');
    elements.btnUpload.classList.add('hidden');
}

// Upload handling
async function handleUpload() {
    if (!state.selectedFile || state.isUploading) return;
    
    state.isUploading = true;
    elements.btnUpload.disabled = true;
    
    showSection('progress');
    updateProgress(0, 'Uploading', 'Uploading your video to the server...');
    
    try {
        // Upload the file
        const formData = new FormData();
        formData.append('file', state.selectedFile);
        
        const response = await fetchWithRetry(`${CONFIG.API_URL}/api/upload`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const data = await response.json();
        state.jobId = data.job_id;
        
        console.log('Upload successful, job ID:', state.jobId);
        
        // Start polling for status
        state.isProcessing = true;
        state.pollAttempts = 0;
        startStatusPolling();
        
    } catch (error) {
        console.error('Upload error:', error);
        showError(error.message || 'Failed to upload video. Please try again.');
    } finally {
        state.isUploading = false;
        elements.btnUpload.disabled = false;
    }
}

// Status polling
function startStatusPolling() {
    if (state.pollInterval) {
        clearInterval(state.pollInterval);
    }
    
    updateProgress(5, 'Processing', 'Detecting faces in your video...');
    
    state.pollInterval = setInterval(async () => {
        if (!state.isProcessing || !state.jobId) {
            stopStatusPolling();
            return;
        }
        
        state.pollAttempts++;
        
        if (state.pollAttempts > CONFIG.MAX_POLL_ATTEMPTS) {
            stopStatusPolling();
            showError('Processing timeout. The video may be too long or complex.');
            return;
        }
        
        try {
            const response = await fetch(`${CONFIG.API_URL}/api/status/${state.jobId}`);
            
            if (!response.ok) {
                throw new Error('Failed to get status');
            }
            
            const data = await response.json();
            
            if (data.status === 'processing' || data.status === 'queued') {
                const progress = Math.max(5, Math.min(95, data.progress));
                const statusText = data.status === 'queued' ? 'Queued' : 'Processing';
                const detailText = getProgressDetail(progress);
                updateProgress(progress, statusText, detailText);
            } else if (data.status === 'complete') {
                stopStatusPolling();
                updateProgress(100, 'Complete', 'All faces have been blurred!');
                setTimeout(() => showSection('complete'), 500);
            } else if (data.status === 'error') {
                stopStatusPolling();
                showError(data.error || 'Processing failed. Please try again.');
            }
            
        } catch (error) {
            console.error('Status poll error:', error);
            // Don't stop polling on transient errors
        }
        
    }, CONFIG.STATUS_POLL_INTERVAL);
}

function stopStatusPolling() {
    if (state.pollInterval) {
        clearInterval(state.pollInterval);
        state.pollInterval = null;
    }
    state.isProcessing = false;
}

function getProgressDetail(progress) {
    if (progress < 10) return 'Preparing video for processing...';
    if (progress < 30) return 'Extracting video frames...';
    if (progress < 70) return 'Detecting and blurring faces...';
    if (progress < 90) return 'Processing remaining frames...';
    return 'Assembling final video...';
}

function updateProgress(percent, status, detail) {
    elements.progressFill.style.width = `${percent}%`;
    elements.progressPercentage.textContent = `${percent}%`;
    elements.progressStatus.textContent = status;
    elements.progressDetail.textContent = detail;
}

// Cancel handling
function handleCancel() {
    stopStatusPolling();
    
    // Cleanup on server if we have a job ID
    if (state.jobId) {
        fetch(`${CONFIG.API_URL}/api/cleanup/${state.jobId}`, {
            method: 'POST',
        }).catch(() => {}); // Ignore errors
    }
    
    resetState();
    showSection('upload');
}

// Download handling
async function handleDownload() {
    if (!state.jobId) return;
    
    elements.btnDownload.disabled = true;
    showLoading(true);
    
    try {
        const response = await fetch(`${CONFIG.API_URL}/api/download/${state.jobId}`);
        
        if (!response.ok) {
            throw new Error('Download failed');
        }
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        // Create download link
        const a = document.createElement('a');
        a.href = url;
        a.download = `blurred_video_${state.jobId.substring(0, 8)}.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        // Cleanup on server
        fetch(`${CONFIG.API_URL}/api/cleanup/${state.jobId}`, {
            method: 'POST',
        }).catch(() => {});
        
    } catch (error) {
        console.error('Download error:', error);
        showError('Failed to download video. Please try again.');
    } finally {
        elements.btnDownload.disabled = false;
        showLoading(false);
    }
}

// New video handling
function handleNewVideo() {
    resetState();
    showSection('upload');
}

// Retry handling
function handleRetry() {
    showSection('upload');
}

// Error handling
function showError(message) {
    elements.errorMessage.textContent = message;
    showSection('error');
}

// Section management
function showSection(sectionName) {
    const sections = ['upload', 'progress', 'complete', 'error'];
    
    sections.forEach(name => {
        const section = elements[`${name}Section`];
        if (section) {
            section.classList.toggle('hidden', name !== sectionName);
        }
    });
}

// State management
function resetState() {
    state.selectedFile = null;
    state.jobId = null;
    state.pollAttempts = 0;
    state.isUploading = false;
    state.isProcessing = false;
    
    stopStatusPolling();
    
    elements.fileInput.value = '';
    elements.filePreview.classList.add('hidden');
    elements.btnUpload.classList.add('hidden');
    elements.progressFill.style.width = '0%';
    elements.progressPercentage.textContent = '0%';
}

// Loading overlay
function showLoading(show) {
    elements.loadingOverlay.classList.toggle('hidden', !show);
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function fetchWithRetry(url, options, attempts = CONFIG.RETRY_ATTEMPTS) {
    for (let i = 0; i < attempts; i++) {
        try {
            const response = await fetch(url, options);
            return response;
        } catch (error) {
            if (i === attempts - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY * (i + 1)));
        }
    }
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CONFIG, state, init };
}
