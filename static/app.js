// Document Q&A AI Agent - Frontend JavaScript

const API_BASE = '/api';

// ============================================
// Utility Functions
// ============================================

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.remove('hidden');
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

function formatTime(seconds) {
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    return `${seconds.toFixed(2)}s`;
}

async function apiRequest(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || data.error || 'API request failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ============================================
// Tab Navigation
// ============================================

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        // Update tabs
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        const tabName = tab.dataset.tab;
        const content = document.getElementById(`${tabName}-tab`);
        if (content) {
            content.classList.add('active');
        }
        
        // Load data for specific tabs
        if (tabName === 'documents') {
            loadDocuments();
        } else if (tabName === 'qa') {
            loadDocumentSelector();
        }
    });
});

// ============================================
// File Upload
// ============================================

const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    handleUpload(files);
});

fileInput.addEventListener('change', () => {
    handleUpload(fileInput.files);
});

async function handleUpload(files) {
    const progressDiv = document.getElementById('upload-progress');
    const progressFill = document.getElementById('progress-fill');
    const statusP = document.getElementById('upload-status');
    const resultDiv = document.getElementById('upload-result');
    
    progressDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showToast(`Skipping ${file.name}: Only PDF files allowed`, 'error');
            continue;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            progressFill.style.width = `${((i + 1) / files.length) * 100}%`;
            statusP.textContent = `Uploading ${file.name}...`;
            
            const data = await fetch(`${API_BASE}/documents/upload`, {
                method: 'POST',
                body: formData
            }).then(r => r.json());
            
            resultDiv.innerHTML = `
                <div class="success-message">
                    <strong>✅ ${data.title}</strong><br>
                    Pages: ${data.page_count} | Size: ${data.file_size}<br>
                    <small>ID: ${data.document_id}</small>
                </div>
            `;
            resultDiv.classList.remove('hidden');
            
            showToast(`Uploaded: ${file.name}`, 'success');
            
        } catch (error) {
            showToast(`Error uploading ${file.name}: ${error.message}`, 'error');
        }
    }
    
    progressDiv.classList.add('hidden');
    progressFill.style.width = '0%';
}

// ============================================
// Documents List
// ============================================

async function loadDocuments() {
    const listDiv = document.getElementById('documents-list');
    const noDocsDiv = document.getElementById('no-docs');
    const refreshBtn = document.getElementById('refresh-docs');
    
    try {
        refreshBtn.disabled = true;
        refreshBtn.textContent = 'Loading...';
        
        const documents = await apiRequest('/documents');
        
        if (documents.length === 0) {
            listDiv.innerHTML = '';
            noDocsDiv.classList.remove('hidden');
            return;
        }
        
        noDocsDiv.classList.add('hidden');
        
        listDiv.innerHTML = documents.map(doc => `
            <div class="document-card" data-id="${doc.document_id}">
                <h4>${escapeHtml(doc.title)}</h4>
                <div class="meta">${doc.page_count} pages</div>
                <div class="actions">
                    <button class="btn btn-secondary btn-small" onclick="viewDoc('${doc.document_id}')">
                        View
                    </button>
                    <button class="btn btn-danger btn-small" onclick="deleteDoc('${doc.document_id}')">
                        Delete
                    </button>
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        showToast('Error loading documents', 'error');
    } finally {
        refreshBtn.disabled = false;
        refreshBtn.textContent = '🔄 Refresh';
    }
}

document.getElementById('refresh-docs').addEventListener('click', loadDocuments);

async function viewDoc(docId) {
    try {
        const doc = await apiRequest(`/documents/${docId}`);
        
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="close" onclick="this.parentElement.parentElement.remove()">&times;</span>
                <h2>${escapeHtml(doc.metadata.title)}</h2>
                <p><strong>Author:</strong> ${escapeHtml(doc.metadata.author || 'Unknown')}</p>
                <p><strong>Pages:</strong> ${doc.metadata.page_count}</p>
                <p><strong>Structure:</strong></p>
                <ul>
                    <li>Has Abstract: ${doc.structure.has_abstract ? 'Yes' : 'No'}</li>
                    <li>Has References: ${doc.structure.has_references ? 'Yes' : 'No'}</li>
                    <li>Sections Found: ${doc.structure.estimated_sections}</li>
                </ul>
                <p><strong>Tables:</strong> ${doc.tables_count}</p>
            </div>
        `;
        document.body.appendChild(modal);
        
    } catch (error) {
        showToast('Error loading document details', 'error');
    }
}

async function deleteDoc(docId) {
    if (!confirm('Are you sure you want to delete this document?')) return;
    
    try {
        await apiRequest(`/documents/${docId}`, { method: 'DELETE' });
        showToast('Document deleted', 'success');
        loadDocuments();
    } catch (error) {
        showToast('Error deleting document', 'error');
    }
}

// ============================================
// Q&A Section
// ============================================

async function loadDocumentSelector() {
    const selectorDiv = document.getElementById('doc-selector');
    
    try {
        const documents = await apiRequest('/documents');
        
        if (documents.length === 0) {
            selectorDiv.innerHTML = '<p class="empty-state">No documents available. Upload some documents first.</p>';
            return;
        }
        
        selectorDiv.innerHTML = documents.map(doc => `
            <label class="doc-checkbox">
                <input type="checkbox" name="selected-doc" value="${doc.document_id}">
                <span>${escapeHtml(doc.title)}</span>
            </label>
        `).join('');
        
        // Select all by default
        document.getElementById('select-all-docs').checked = true;
        
    } catch (error) {
        showToast('Error loading documents', 'error');
    }
}

// Select all checkbox
document.getElementById('select-all-docs').addEventListener('change', function() {
    document.querySelectorAll('input[name="selected-doc"]').forEach(cb => {
        cb.checked = this.checked;
    });
});

// Ask question
document.getElementById('ask-btn').addEventListener('click', askQuestion);

async function askQuestion() {
    const questionInput = document.getElementById('question-input');
    const question = questionInput.value.trim();
    
    if (!question) {
        showToast('Please enter a question', 'error');
        return;
    }
    
    // Get selected documents
    const selectedDocs = Array.from(document.querySelectorAll('input[name="selected-doc"]:checked'))
        .map(cb => cb.value);
    
    if (selectedDocs.length === 0) {
        showToast('Please select at least one document', 'error');
        return;
    }
    
    const loadingDiv = document.getElementById('loading');
    const answerSection = document.getElementById('answer-section');
    const answerContent = document.getElementById('answer-content');
    const answerMeta = document.getElementById('answer-meta');
    
    loadingDiv.classList.remove('hidden');
    answerSection.classList.add('hidden');
    
    try {
        const response = await apiRequest('/qa/query', {
            method: 'POST',
            body: JSON.stringify({
                question: question,
                document_ids: selectedDocs,
                use_cache: true
            })
        });
        
        loadingDiv.classList.add('hidden');
        answerSection.classList.remove('hidden');
        
        // Format answer
        let formattedAnswer = escapeHtml(response.answer);
        formattedAnswer = formattedAnswer.replace(/\n/g, '<br>');
        formattedAnswer = formattedAnswer.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedAnswer = formattedAnswer.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        answerContent.innerHTML = formattedAnswer;
        
        // Meta info
        let metaHtml = `Type: ${response.query_type} | `;
        metaHtml += `Tokens: ${response.tokens_used} | `;
        metaHtml += `Time: ${formatTime(response.response_time)}`;
        
        if (response.cached) {
            metaHtml += ' | 📦 Cached';
        }
        
        answerMeta.innerHTML = metaHtml;
        
    } catch (error) {
        loadingDiv.classList.add('hidden');
        showToast('Error processing question: ' + error.message, 'error');
    }
}

// Enter key to submit question
document.getElementById('question-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        askQuestion();
    }
});

// ============================================
// ArXiv Section
// ============================================

document.getElementById('arxiv-search-btn').addEventListener('click', () => {
    searchArxiv();
});

document.getElementById('arxiv-desc-btn').addEventListener('click', () => {
    searchArxivByDescription();
});

async function searchArxiv() {
    const query = document.getElementById('arxiv-query').value.trim();
    const maxResults = parseInt(document.getElementById('arxiv-max').value) || 5;
    
    if (!query) {
        showToast('Please enter a search query', 'error');
        return;
    }
    
    try {
        const data = await apiRequest('/arxiv/search', {
            method: 'POST',
            body: JSON.stringify({
                query: query,
                max_results: maxResults,
                sort_by: 'relevance',
                sort_order: 'descending'
            })
        });
        
        displayArxivResults(data.papers);
        
    } catch (error) {
        showToast('Error searching ArXiv: ' + error.message, 'error');
    }
}

async function searchArxivByDescription() {
    const description = document.getElementById('arxiv-desc').value.trim();
    
    if (!description) {
        showToast('Please enter a description', 'error');
        return;
    }
    
    try {
        const data = await apiRequest('/arxiv/search-by-description', {
            method: 'POST',
            body: JSON.stringify({
                query: description,
                max_results: 5
            })
        });
        
        displayArxivResults(data.papers);
        
    } catch (error) {
        showToast('Error searching ArXiv: ' + error.message, 'error');
    }
}

function displayArxivResults(papers) {
    const resultsDiv = document.getElementById('arxiv-results');
    
    if (!papers || papers.length === 0) {
        resultsDiv.innerHTML = '<p class="empty-state">No papers found.</p>';
        return;
    }
    
    resultsDiv.innerHTML = papers.map(paper => `
        <div class="arxiv-card">
            <h4>${escapeHtml(paper.title)}</h4>
            <div class="authors">${escapeHtml(paper.authors.slice(0, 3).join(', '))}${paper.authors.length > 3 ? ' et al.' : ''}</div>
            <div class="abstract">${escapeHtml(paper.abstract)}</div>
            <div class="tags">
                <span class="tag">${escapeHtml(paper.primary_category)}</span>
                <span class="tag">${escapeHtml(paper.published_date)}</span>
            </div>
            <div class="actions" style="margin-top: 12px;">
                <a href="${paper.pdf_url}" target="_blank" class="btn btn-secondary btn-small">
                    📄 View PDF
                </a>
                <button class="btn btn-primary btn-small" onclick="ingestArxiv('${escapeHtml(paper.arxiv_id)}')">
                    📥 Ingest
                </button>
            </div>
        </div>
    `).join('');
}

async function ingestArxiv(arxivId) {
    try {
        await apiRequest('/arxiv/ingest', {
            method: 'POST',
            body: JSON.stringify({ arxiv_id: arxivId })
        });
        
        showToast(`Paper ${arxivId} ingested successfully!`, 'success');
        
    } catch (error) {
        showToast('Error ingesting paper: ' + error.message, 'error');
    }
}

// ============================================
// Utility Functions
// ============================================

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Load documents on init
    loadDocuments();
    loadDocumentSelector();
    
    // Check API health
    apiRequest('/health').then(data => {
        console.log('API Health:', data);
    }).catch(error => {
        console.error('API not available:', error);
        showToast('API not available. Make sure the server is running.', 'error');
    });
});

// Modal styles (injected dynamically)
const style = document.createElement('style');
style.textContent = `
    .modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }
    .modal-content {
        background: white;
        padding: 24px;
        border-radius: 12px;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
        position: relative;
    }
    .modal-content .close {
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 24px;
        cursor: pointer;
    }
    .modal-content ul {
        margin-left: 20px;
    }
    .modal-content li {
        margin: 4px 0;
    }
    .success-message {
        background: #dcfce7;
        padding: 16px;
        border-radius: 8px;
        margin-top: 16px;
    }
`;
document.head.appendChild(style);

