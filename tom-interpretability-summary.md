# A Sparse ToM Circuit in Gemma 2-2B

<div id="pdf-container" style="background-color: white; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin: 20px 0; border-radius: 4px; overflow: hidden;">
  <div id="pdf-controls" style="background-color: white; padding: 10px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f0f0f0;">
    <div>
      <span id="page-count-info" style="margin: 0 15px; font-size: 14px;">Total pages: <span id="page-count">...</span></span>
    </div>
    <div>
      <button id="zoom-out" style="background-color: white; border: 1px solid #ddd; padding: 8px 15px; margin: 0 5px; border-radius: 4px; cursor: pointer; font-size: 14px;">Zoom Out</button>
      <button id="zoom-in" style="background-color: white; border: 1px solid #ddd; padding: 8px 15px; margin: 0 5px; border-radius: 4px; cursor: pointer; font-size: 14px;">Zoom In</button>
    </div>
  </div>
  <div id="pdf-viewer" style="padding: 20px; overflow: auto; max-height: 800px;">
    <div id="loading-message" style="text-align: center; padding: 20px; font-style: italic; color: #666;">Loading PDF...</div>
    <div id="pages-container"></div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<script>
  // Configure PDF.js worker
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  
  // PDF file path - adjust as needed
  const pdfUrl = '/pages/document.pdf';
  
  // Variables to store the current state
  let pdfDoc = null;
  let scale = 1.5; // Initial zoom level
  let currentPages = [];
  
  // Get page elements
  const pagesContainer = document.getElementById('pages-container');
  const zoomInButton = document.getElementById('zoom-in');
  const zoomOutButton = document.getElementById('zoom-out');
  const pageCountSpan = document.getElementById('page-count');
  const loadingMessage = document.getElementById('loading-message');
  
  // Render all PDF pages
  async function renderAllPages() {
    // Clear existing pages
    pagesContainer.innerHTML = '';
    currentPages = [];
    
    // Show loading message
    loadingMessage.style.display = 'block';
    
    // Render each page
    for (let i = 1; i <= pdfDoc.numPages; i++) {
      const pageDiv = document.createElement('div');
      pageDiv.style.marginBottom = '20px';
      pageDiv.style.display = 'flex';
      pageDiv.style.justifyContent = 'center';
      
      const canvas = document.createElement('canvas');
      canvas.style.maxWidth = '100%';
      canvas.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
      pageDiv.appendChild(canvas);
      pagesContainer.appendChild(pageDiv);
      
      // Get and render the page
      const page = await pdfDoc.getPage(i);
      const viewport = page.getViewport({ scale: scale });
      
      canvas.height = viewport.height;
      canvas.width = viewport.width;
      
      const ctx = canvas.getContext('2d');
      await page.render({
        canvasContext: ctx,
        viewport: viewport
      }).promise;
      
      currentPages.push({ pageDiv, canvas, page });
    }
    
    // Hide loading message
    loadingMessage.style.display = 'none';
  }
  
  // Zoom functionality - rerender all pages at new scale
  async function zoomIn() {
    scale += 0.25;
    await renderAllPages();
  }
  
  async function zoomOut() {
    if (scale <= 0.5) return; // Prevent zooming out too much
    scale -= 0.25;
    await renderAllPages();
  }
  
  // Add event listeners
  zoomInButton.addEventListener('click', zoomIn);
  zoomOutButton.addEventListener('click', zoomOut);
  
  // Load the PDF
  pdfjsLib.getDocument(pdfUrl).promise.then(function(pdf) {
    pdfDoc = pdf;
    pageCountSpan.textContent = pdf.numPages;
    
    // Render all pages
    renderAllPages();
  }).catch(function(error) {
    loadingMessage.textContent = 'Error loading PDF: ' + error.message;
  });
</script>
