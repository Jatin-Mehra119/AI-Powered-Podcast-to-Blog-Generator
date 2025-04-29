document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const uploadForm = document.getElementById('upload-form');
    const audioFileInput = document.getElementById('audio-file');
    const uploadSection = document.getElementById('upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');
    const processingStatus = document.getElementById('processing-status');
    const filesContainer = document.getElementById('files-container');
    const generateNewBtn = document.getElementById('generate-new');
    const tryAgainBtn = document.getElementById('try-again');
    const fileLabel = document.querySelector('.file-label');

    // Selected file name display
    audioFileInput.addEventListener('change', () => {
        if (audioFileInput.files.length > 0) {
            fileLabel.textContent = audioFileInput.files[0].name;
        } else {
            fileLabel.textContent = 'Choose a file';
        }
    });

    // Form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Validate file
        const file = audioFileInput.files[0];
        if (!file) {
            showError('Please select an audio file.');
            return;
        }

        // Check file size (max 20MB)
        if (file.size > 20 * 1024 * 1024) {
            showError('File size exceeds the 20MB limit.');
            return;
        }

        // Check file type
        const validTypes = ['.mp3', '.wav', '.m4a', '.ogg'];
        const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        if (!validTypes.includes(fileExtension)) {
            showError('Only .mp3, .wav, .m4a, and .ogg files are supported.');
            return;
        }

        // Get selected content types
        const contentTypes = Array.from(
            document.querySelectorAll('input[name="content_types"]:checked')
        ).map(checkbox => checkbox.value);

        if (contentTypes.length === 0) {
            showError('Please select at least one content type to generate.');
            return;
        }

        // Show processing section
        uploadSection.classList.add('hidden');
        processingSection.classList.remove('hidden');

        // Create FormData
        const formData = new FormData();
        formData.append('file', file);
        contentTypes.forEach(type => {
            formData.append('content_types', type);
        });

        try {
            // Upload file
            const uploadResponse = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const errorData = await uploadResponse.json();
                throw new Error(errorData.error || 'Error uploading file');
            }

            const uploadData = await uploadResponse.json();
            const jobId = uploadData.job_id;
            
            // Poll job status
            await pollJobStatus(jobId);
        } catch (error) {
            console.error('Error:', error);
            showError(error.message || 'An unexpected error occurred.');
        }
    });

    // Function to poll job status
    async function pollJobStatus(jobId) {
        const maxRetries = 60; // 5 minutes (5s intervals)
        let retryCount = 0;
        
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/api/status/${jobId}`);
                if (!response.ok) {
                    throw new Error('Error checking job status');
                }
                
                const data = await response.json();
                
                // Update status message based on job status
                if (data.status === 'processing') {
                    processingStatus.textContent = `Processing your audio file: ${data.filename}`;
                } else if (data.status === 'completed') {
                    clearInterval(interval);
                    showResults(data.files);
                } else if (data.status === 'failed') {
                    clearInterval(interval);
                    showError(data.error || 'Processing failed');
                }
            } catch (error) {
                console.error('Error polling job status:', error);
                retryCount++;
                
                if (retryCount >= maxRetries) {
                    clearInterval(interval);
                    showError('Processing timed out. Please try again later.');
                }
            }
        }, 5000); // Poll every 5 seconds
    }

    // Function to display results
    function showResults(files) {
        processingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        // Clear any previous results
        filesContainer.innerHTML = '';
        
        // Display files for download
        const fileTypes = {
            'transcript': 'Transcript',
            'blog': 'Blog Post',
            'seo': 'SEO Elements',
            'faq': 'FAQ Section',
            'social': 'Social Media Posts',
            'newsletter': 'Newsletter',
            'quotes': 'Quotes'
        };
        
        // Sort files by type for consistent display order
        const sortOrder = ['blog', 'transcript', 'seo', 'faq', 'social', 'newsletter', 'quotes'];
        const fileEntries = Object.entries(files);
        
        fileEntries.sort((a, b) => {
            const typeA = a[0].split('_').pop(); // Get the type part of the filename
            const typeB = b[0].split('_').pop();
            return sortOrder.indexOf(typeA) - sortOrder.indexOf(typeB);
        });
        
        fileEntries.forEach(([key, filename]) => {
            const fileType = key.split('_').pop();
            const displayName = fileTypes[fileType] || fileType;
            const fileCard = document.createElement('div');
            fileCard.className = 'file-card';
            
            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-info';
            
            const fileTitle = document.createElement('h3');
            fileTitle.textContent = displayName;
            
            const fileDescription = document.createElement('p');
            fileDescription.textContent = `Generated from your audio`;
            
            const downloadButton = document.createElement('button');
            downloadButton.className = 'file-btn';
            downloadButton.textContent = 'Download';
            downloadButton.addEventListener('click', () => {
                window.location.href = `/api/download/${filename}`;
            });
            
            fileInfo.appendChild(fileTitle);
            fileInfo.appendChild(fileDescription);
            fileCard.appendChild(fileInfo);
            fileCard.appendChild(downloadButton);
            filesContainer.appendChild(fileCard);
        });
    }

    // Function to show error message
    function showError(message) {
        uploadSection.classList.add('hidden');
        processingSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.remove('hidden');
        
        errorMessage.textContent = message;
    }

    // Event listeners for buttons
    generateNewBtn.addEventListener('click', resetForm);
    tryAgainBtn.addEventListener('click', resetForm);

    // Function to reset form and UI
    function resetForm() {
        uploadForm.reset();
        fileLabel.textContent = 'Choose a file';
        
        uploadSection.classList.remove('hidden');
        processingSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
    }

    // File drag and drop functionality
    const uploadContainer = document.querySelector('.upload-container');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadContainer.classList.add('highlight');
    }
    
    function unhighlight() {
        uploadContainer.classList.remove('highlight');
    }
    
    uploadContainer.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            audioFileInput.files = files;
            
            // Trigger change event to update UI
            const event = new Event('change');
            audioFileInput.dispatchEvent(event);
        }
    }
});