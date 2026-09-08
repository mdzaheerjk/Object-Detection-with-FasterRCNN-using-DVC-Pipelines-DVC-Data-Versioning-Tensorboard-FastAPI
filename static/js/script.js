/* ==========================================================================
   Guns Object Detection Studio - Interactive Client Logic
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const detectBtn = document.getElementById('detectBtn');
    const confidenceSlider = document.getElementById('confidenceSlider');
    const confidenceVal = document.getElementById('confidenceVal');
    
    const emptyState = document.getElementById('emptyState');
    const imageContainer = document.getElementById('imageContainer');
    const resultImage = document.getElementById('resultImage');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    
    const tabOriginal = document.getElementById('tabOriginal');
    const tabDetected = document.getElementById('tabDetected');
    
    const metricCount = document.getElementById('metricCount');
    const metricConfidence = document.getElementById('metricConfidence');
    const metricLatency = document.getElementById('metricLatency');
    
    const downloadBtn = document.getElementById('downloadBtn');
    const resetBtn = document.getElementById('resetBtn');
    const presetButtons = document.querySelectorAll('.preset-btn');

    let currentFile = null;
    let originalImageUrl = null;
    let detectedImageUrl = null;

    // 1. Confidence Slider Listener
    confidenceSlider.addEventListener('input', (e) => {
        confidenceVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // 2. Drag & Drop Handling
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // 3. Clipboard Paste
    window.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                handleFileSelection(blob);
                break;
            }
        }
    });

    // 4. Handle Preset Buttons
    presetButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            const imgSrc = btn.getAttribute('data-src');
            if (!imgSrc) return;
            try {
                showLoading('Loading preset image...');
                const response = await fetch(imgSrc);
                const blob = await response.blob();
                const file = new File([blob], 'sample.jpg', { type: blob.type });
                handleFileSelection(file);
                // Auto trigger detection on preset click
                setTimeout(() => runDetection(), 200);
            } catch (err) {
                console.error('Failed to load preset:', err);
                hideLoading();
            }
        });
    });

    // 5. Handle File Selection
    function handleFileSelection(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (PNG, JPG, JPEG).');
            return;
        }

        currentFile = file;
        originalImageUrl = URL.createObjectURL(file);
        detectedImageUrl = null;

        // Display preview
        emptyState.style.display = 'none';
        imageContainer.style.display = 'flex';
        resultImage.src = originalImageUrl;

        tabOriginal.classList.add('active');
        tabDetected.classList.remove('active');

        detectBtn.disabled = false;
        downloadBtn.style.display = 'none';

        metricCount.textContent = '-';
        metricConfidence.textContent = '-';
        metricLatency.textContent = '-';
    }

    // 6. View Tabs
    tabOriginal.addEventListener('click', () => {
        if (originalImageUrl) {
            resultImage.src = originalImageUrl;
            tabOriginal.classList.add('active');
            tabDetected.classList.remove('active');
        }
    });

    tabDetected.addEventListener('click', () => {
        if (detectedImageUrl) {
            resultImage.src = detectedImageUrl;
            tabDetected.classList.add('active');
            tabOriginal.classList.remove('active');
        }
    });

    // 7. Run Detection
    detectBtn.addEventListener('click', runDetection);

    async function runDetection() {
        if (!currentFile) return;

        const confidence = confidenceSlider.value;
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('confidence', confidence);

        showLoading('Running Faster R-CNN inference...');
        const startTime = performance.now();

        try {
            // First call /api/detect/ for structured stats (if available)
            let detectionStats = null;
            try {
                const statsResponse = await fetch(`/api/detect/?confidence=${confidence}`, {
                    method: 'POST',
                    body: formData
                });
                if (statsResponse.ok) {
                    detectionStats = await statsResponse.json();
                }
            } catch (e) {
                console.warn('Stats endpoint unavailable, fallback to stream.');
            }

            // Stream processed image from /predict/
            const predictResponse = await fetch(`/predict/?confidence=${confidence}`, {
                method: 'POST',
                body: formData
            });

            if (!predictResponse.ok) {
                throw new Error(`Server returned status: ${predictResponse.status}`);
            }

            const imageBlob = await predictResponse.blob();
            detectedImageUrl = URL.createObjectURL(imageBlob);

            const endTime = performance.now();
            const latencyMs = Math.round(endTime - startTime);

            // Update UI
            resultImage.src = detectedImageUrl;
            tabDetected.classList.add('active');
            tabOriginal.classList.remove('active');

            downloadBtn.style.display = 'inline-flex';

            if (detectionStats) {
                metricCount.textContent = detectionStats.count;
                metricConfidence.textContent = detectionStats.max_score ? `${(detectionStats.max_score * 100).toFixed(1)}%` : '0%';
                metricLatency.textContent = `${detectionStats.latency_ms || latencyMs} ms`;
            } else {
                metricLatency.textContent = `${latencyMs} ms`;
                metricCount.textContent = 'Detected';
                metricConfidence.textContent = `${Math.round(confidence * 100)}%+`;
            }

        } catch (err) {
            console.error('Detection error:', err);
            alert(`Detection failed: ${err.message}`);
        } finally {
            hideLoading();
        }
    }

    // 8. Download Result
    downloadBtn.addEventListener('click', () => {
        if (!detectedImageUrl) return;
        const link = document.createElement('a');
        link.href = detectedImageUrl;
        link.download = `detected_guns_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // 9. Reset
    resetBtn.addEventListener('click', () => {
        currentFile = null;
        originalImageUrl = null;
        detectedImageUrl = null;
        fileInput.value = '';

        imageContainer.style.display = 'none';
        emptyState.style.display = 'block';
        detectBtn.disabled = true;
        downloadBtn.style.display = 'none';

        metricCount.textContent = '-';
        metricConfidence.textContent = '-';
        metricLatency.textContent = '-';
    });

    function showLoading(text = 'Processing...') {
        loadingText.textContent = text;
        loadingOverlay.style.display = 'flex';
    }

    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }
});
