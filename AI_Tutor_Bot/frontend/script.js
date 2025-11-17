let currentTopicId = null;
const API_BASE = 'http://localhost:8000';

async function uploadPDF() {
    const fileInput = document.getElementById('pdfUpload');
    const statusDiv = document.getElementById('uploadStatus');
    
    if (!fileInput.files.length) {
        statusDiv.innerHTML = '<span style="color: red;">Please select a PDF file</span>';
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    statusDiv.innerHTML = '<span style="color: blue;">Processing PDF... This may take a moment.</span>';

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        currentTopicId = data.topic_id;
        
        statusDiv.innerHTML = `<span style="color: green;">${data.message}. ${data.chunks_created} text chunks created.</span>`;
        
        // Show chat section
        document.getElementById('chatSection').style.display = 'block';
        
        // Add welcome message
        addAIMessage("Hello! I'm your AI tutor. I've processed the PDF and I'm ready to answer your questions about the chapter. What would you like to know?");
        
    } catch (error) {
        console.error('Upload error:', error);
        statusDiv.innerHTML = `<span style="color: red;">Error uploading file: ${error.message}</span>`;
    }
}

async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();
    
    if (!message || !currentTopicId) return;

    // Add user message to chat
    addUserMessage(message);
    userInput.value = '';

    // Show loading
    addAIMessage('Thinking...', true);

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                topic_id: currentTopicId,
                question: message
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Remove loading message
        removeLoadingMessage();
        
        // Add AI response with image
        addAIMessage(data.answer, false, data.image_filename, data.image_title, data.sources);
        
    } catch (error) {
        console.error('Chat error:', error);
        removeLoadingMessage();
        addAIMessage("I'm sorry, I encountered an error while processing your question. Please try again.");
    }
}

function addUserMessage(message) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addAIMessage(message, isLoading = false, imageFilename = null, imageTitle = null, sources = []) {
    const chatMessages = document.getElementById('chatMessages');
    
    if (isLoading) {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message ai-message loading';
        loadingDiv.id = 'loadingMessage';
        loadingDiv.textContent = message;
        chatMessages.appendChild(loadingDiv);
    } else {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai-message';
        
        // Add answer text
        const textDiv = document.createElement('div');
        textDiv.innerHTML = message.replace(/\n/g, '<br>');
        messageDiv.appendChild(textDiv);
        
        // Add image if available
        if (imageFilename) {
            const img = document.createElement('img');
            img.src = `${API_BASE}/images/file/${imageFilename}`;
            img.alt = imageTitle || 'Relevant diagram';
            messageDiv.appendChild(img);
            
            if (imageTitle) {
                const caption = document.createElement('div');
                caption.className = 'image-caption';
                caption.textContent = imageTitle;
                messageDiv.appendChild(caption);
            }
        }
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            sourcesDiv.textContent = `Sources: ${sources.join(', ')}`;
            messageDiv.appendChild(sourcesDiv);
        }
        
        chatMessages.appendChild(messageDiv);
    }
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeLoadingMessage() {
    const loadingMessage = document.getElementById('loadingMessage');
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('AI Tutor Chatbot initialized');
});