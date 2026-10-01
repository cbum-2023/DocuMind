const API = "http://localhost:8000";

let documents = [];
let selectedDocId = null;
let selectedDocName = null;
let chatHistory = [];

document.addEventListener("DOMContentLoaded", () => {
  initDropZone();
  fetchDocuments();
  checkHealth();

  document.getElementById("clearAllBtn").addEventListener("click", clearAllDocuments);
  document.getElementById("resetFilterBtn").addEventListener("click", resetDocFilter);
});

async function checkHealth() {
  const statusEl = document.getElementById("serverStatus");
  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      statusEl.innerHTML = `<span class="status-pulse"></span> Backend Connected`;
    } else {
      statusEl.innerHTML = `<span class="dot" style="background:var(--danger)"></span> Service Degraded`;
    }
  } catch (e) {
    statusEl.innerHTML = `<span class="dot" style="background:var(--danger)"></span> Offline`;
  }
}

// Drag & Drop Upload Handlers
function initDropZone() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
      uploadFile(e.target.files[0]);
    }
  });
}

async function uploadFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    alert("Please select a PDF file.");
    return;
  }

  const progressEl = document.getElementById("uploadProgress");
  const statusText = document.getElementById("uploadStatusText");
  progressEl.classList.remove("hidden");
  statusText.textContent = `Uploading ${file.name}...`;

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");

    statusText.textContent = `Success! Saved ${data.chunks} chunks.`;
    setTimeout(() => progressEl.classList.add("hidden"), 2000);
    await fetchDocuments();
  } catch (e) {
    alert("Upload Error: " + e.message);
    progressEl.classList.add("hidden");
  }
}

// Document Inventory Management
async function fetchDocuments() {
  try {
    const res = await fetch(`${API}/documents`);
    const data = await res.json();
    documents = data.documents || [];
    renderDocumentList();
  } catch (e) {
    console.error("Failed to fetch documents", e);
  }
}

function renderDocumentList() {
  const listEl = document.getElementById("documentList");
  const countEl = document.getElementById("docCount");

  countEl.textContent = documents.length;

  if (documents.length === 0) {
    listEl.innerHTML = `<div class="empty-state">No PDFs uploaded yet.</div>`;
    resetDocFilter();
    return;
  }

  listEl.innerHTML = "";
  documents.forEach((doc) => {
    const item = document.createElement("div");
    item.className = `doc-item ${selectedDocId === doc.doc_id ? "active" : ""}`;
    item.onclick = (e) => {
      if (e.target.classList.contains("delete-doc-btn")) return;
      selectDocFilter(doc.doc_id, doc.filename);
    };

    item.innerHTML = `
      <div class="doc-info">
        <span class="doc-name" title="${escapeHtml(doc.filename)}">📄 ${escapeHtml(doc.filename)}</span>
        <span class="doc-meta">${doc.chunks} chunks</span>
      </div>
      <button class="delete-doc-btn" onclick="deleteDocument('${doc.doc_id}', event)" title="Delete document">🗑</button>
    `;
    listEl.appendChild(item);
  });
}

function selectDocFilter(docId, filename) {
  if (selectedDocId === docId) {
    resetDocFilter();
    return;
  }
  selectedDocId = docId;
  selectedDocName = filename;

  const indicator = document.getElementById("filterIndicator");
  const textEl = document.getElementById("filterText");
  const resetBtn = document.getElementById("resetFilterBtn");

  indicator.innerHTML = `<span class="dot blue"></span> Filtering: <strong>${escapeHtml(filename)}</strong>`;
  indicator.appendChild(resetBtn);
  resetBtn.classList.remove("hidden");

  renderDocumentList();
}

function resetDocFilter() {
  selectedDocId = null;
  selectedDocName = null;

  const indicator = document.getElementById("filterIndicator");
  indicator.innerHTML = `<span class="dot green"></span> Searching across all documents`;
  
  renderDocumentList();
}

async function deleteDocument(docId, event) {
  if (event) event.stopPropagation();
  if (!confirm("Are you sure you want to delete this document?")) return;

  try {
    const res = await fetch(`${API}/documents/${docId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Deletion failed");
    if (selectedDocId === docId) resetDocFilter();
    await fetchDocuments();
  } catch (e) {
    alert("Error deleting document: " + e.message);
  }
}

async function clearAllDocuments() {
  if (!confirm("Are you sure you want to clear ALL uploaded documents?")) return;

  try {
    const res = await fetch(`${API}/documents`, { method: "DELETE" });
    if (!res.ok) throw new Error("Clear all failed");
    resetDocFilter();
    await fetchDocuments();
  } catch (e) {
    alert("Error clearing documents: " + e.message);
  }
}

// Chat Handlers
function usePrompt(text) {
  const input = document.getElementById("questionInput");
  input.value = text;
  input.focus();
}

function handleKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend(e);
  }
}

async function handleSend(e) {
  if (e) e.preventDefault();
  const input = document.getElementById("questionInput");
  const q = input.value.trim();

  if (!q) return;

  // Hide welcome banner on first question
  const welcomeBanner = document.querySelector(".welcome-banner");
  if (welcomeBanner) welcomeBanner.remove();

  // Add User Message
  appendMessage("user", q);
  chatHistory.push({ role: "user", content: q });
  input.value = "";

  // Append Thinking Indicator
  const thinkingId = appendThinkingMessage();
  scrollToBottom();

  try {
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        doc_id: selectedDocId,
        history: chatHistory.slice(-6)
      }),
    });

    const data = await res.json();
    removeThinkingMessage(thinkingId);

    if (!res.ok) throw new Error(data.detail || "Error asking question.");

    chatHistory.push({ role: "assistant", content: data.answer });
    appendMessage("assistant", data.answer, data.sources);
    scrollToBottom();
  } catch (e) {
    removeThinkingMessage(thinkingId);
    appendMessage("assistant", `⚠️ **Error**: ${e.message}`, []);
    scrollToBottom();
  }
}

function appendMessage(role, text, sources = []) {
  const stream = document.getElementById("chatStream");
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatarText = role === "user" ? "👤" : "⚡";

  let sourcesHtml = "";
  if (sources && sources.length > 0) {
    const sourcesListId = `sources-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`;
    let cardsHtml = "";
    sources.forEach((s) => {
      cardsHtml += `
        <div class="source-card">
          <div class="source-header">
            <span>📄 ${escapeHtml(s.filename)}</span>
            <span>Page ${s.page}</span>
          </div>
          <div class="source-text">${escapeHtml(s.text)}</div>
        </div>
      `;
    });

    sourcesHtml = `
      <div class="sources-container">
        <button class="sources-toggle" onclick="toggleSources('${sourcesListId}')">
          📚 Referenced Sources (${sources.length}) ▼
        </button>
        <div id="${sourcesListId}" class="sources-list hidden">
          ${cardsHtml}
        </div>
      </div>
    `;
  }

  // Safe markdown parsing using marked + DOMPurify
  const parsedContent = role === "assistant" 
    ? DOMPurify.sanitize(marked.parse(text))
    : escapeHtml(text);

  row.innerHTML = `
    ${role === "assistant" ? `<div class="avatar">${avatarText}</div>` : ""}
    <div class="msg-content">
      <div class="msg-text">${parsedContent}</div>
      ${sourcesHtml}
    </div>
    ${role === "user" ? `<div class="avatar">${avatarText}</div>` : ""}
  `;

  stream.appendChild(row);
}

function toggleSources(id) {
  const el = document.getElementById(id);
  if (el) {
    el.classList.toggle("hidden");
  }
}

function appendThinkingMessage() {
  const stream = document.getElementById("chatStream");
  const row = document.createElement("div");
  const id = `thinking-${Date.now()}`;
  row.id = id;
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="avatar">⚡</div>
    <div class="msg-content">
      <div class="upload-progress" style="margin-top:0">
        <div class="spinner"></div>
        <span>Thinking & analyzing context...</span>
      </div>
    </div>
  `;
  stream.appendChild(row);
  return id;
}

function removeThinkingMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom() {
  const stream = document.getElementById("chatStream");
  stream.scrollTop = stream.scrollHeight;
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

