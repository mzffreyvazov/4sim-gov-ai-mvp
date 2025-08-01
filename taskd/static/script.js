// Global variables
let currentData = null;

// DOM elements
const uploadForm = document.getElementById('upload-form');
const editSection = document.getElementById('edit-section');
const slidesContainer = document.getElementById('slides-container');
const generatePptxBtn = document.getElementById('generate-pptx-btn');
const generateBtn = document.getElementById('generate-btn');

// Status message functions
function showMessage(message, type = 'info') {
    const statusElement = document.getElementById('status-message');
    statusElement.textContent = message;
    statusElement.className = `status-message ${type}`;
    statusElement.style.display = 'block';
    
    setTimeout(() => {
        statusElement.style.display = 'none';
    }, 5000);
}

// Loading state functions
function setButtonLoading(button, isLoading) {
    const textSpan = button.querySelector('.btn-text');
    const loadingSpan = button.querySelector('.btn-loading');
    
    if (isLoading) {
        textSpan.style.display = 'none';
        loadingSpan.style.display = 'flex';
        button.disabled = true;
    } else {
        textSpan.style.display = 'flex';
        loadingSpan.style.display = 'none';
        button.disabled = false;
    }
}

// Upload form handler
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Preserve form values
    const fileInput = document.getElementById('file-input');
    const promptInput = document.getElementById('prompt-input');
    const slideCountInput = document.getElementById('slide-count-input');
    
    const formValues = {
        prompt: promptInput.value,
        slideCount: slideCountInput.value
    };
    
    setButtonLoading(generateBtn, true);
    showMessage('Uploading document and generating content...', 'info');
    
    const formData = new FormData(uploadForm);
    
    try {
        const response = await fetch('/generate_json/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentData = result.json_data;
            showMessage('Content generated successfully! You can now edit the slides.', 'success');
            displaySlides(currentData);
            editSection.style.display = 'block';
            editSection.scrollIntoView({ behavior: 'smooth' });
            
            // Restore form values
            promptInput.value = formValues.prompt;
            slideCountInput.value = formValues.slideCount;
        } else {
            throw new Error(result.detail || 'Failed to generate content');
        }
    } catch (error) {
        console.error('Error:', error);
        showMessage(`Error: ${error.message}`, 'error');
        
        // Restore form values even on error
        promptInput.value = formValues.prompt;
        slideCountInput.value = formValues.slideCount;
    } finally {
        setButtonLoading(generateBtn, false);
    }
});

// Display slides function
function displaySlides(data) {
    slidesContainer.innerHTML = '';
    
    const presentation = data.presentation;
    let slideNumber = 1;
    
    // 1. Intro Slide
    if (presentation.intro_slide) {
        const introSlide = createSlideCard(
            slideNumber++,
            '🎬 Intro Slide',
            createIntroSlideContent(presentation.intro_slide)
        );
        slidesContainer.appendChild(introSlide);
    }
    
    // 2. Project Content Slide
    if (presentation.project_content_slide) {
        const contentSlide = createSlideCard(
            slideNumber++,
            '📋 Project Overview',
            createProjectContentSlideContent(presentation.project_content_slide)
        );
        slidesContainer.appendChild(contentSlide);
    }
    
    // 3. Content Slides
    if (presentation.content_slides) {
        presentation.content_slides.forEach((slide, index) => {
            const contentSlide = createSlideCard(
                slideNumber++,
                `📋 Content Slide ${index + 1}`,
                createContentSlideContent(slide, index)
            );
            slidesContainer.appendChild(contentSlide);
        });
    }
    
    // 4. Chart Slides
    if (presentation.chart_slides) {
        presentation.chart_slides.forEach((slide, index) => {
            const chartSlide = createSlideCard(
                slideNumber++,
                `📊 Chart Slide ${index + 1}`,
                createChartSlideContent(slide, index)
            );
            slidesContainer.appendChild(chartSlide);
        });
    }
    
    // 5. Final Slide
    if (presentation.final_slide) {
        const finalSlide = createSlideCard(
            slideNumber++,
            '🎯 Next Steps',
            createFinalSlideContent(presentation.final_slide)
        );
        slidesContainer.appendChild(finalSlide);
    }
}

// Create slide card
function createSlideCard(number, title, content) {
    const card = document.createElement('div');
    card.className = 'slide-card';
    
    card.innerHTML = `
        <h3>
            <span class="slide-number">${number}</span>
            ${title}
        </h3>
        ${content}
    `;
    
    return card;
}

// Create intro slide content
function createIntroSlideContent(data) {
    return `
        <div>
            <label><strong>Presentation Title:</strong></label>
            <div class="editable title" data-path="presentation.intro_slide.presentation_title" contenteditable="true">${data.presentation_title || ''}</div>
        </div>
        <div>
            <label><strong>Date:</strong></label>
            <div class="editable date" data-path="presentation.intro_slide.presentation_date" contenteditable="true">${data.presentation_date || ''}</div>
        </div>
    `;
}

// Create project content slide content
function createProjectContentSlideContent(data) {
    return `
        <div>
            <label><strong>Presentation Overview:</strong></label>
            <div class="editable overview" data-path="presentation.project_content_slide.presentation_overview" contenteditable="true">${data.presentation_overview || ''}</div>
        </div>
        <div>
            <label><strong>Presentation Goal:</strong></label>
            <div class="editable overview" data-path="presentation.project_content_slide.presentation_goal" contenteditable="true">${data.presentation_goal || ''}</div>
        </div>
    `;
}

// Create content slide content
function createContentSlideContent(data, slideIndex) {
    let contentHtml = `
        <div>
            <label><strong>Section Title:</strong></label>
            <div class="editable title" data-path="presentation.content_slides.${slideIndex}.general_content_title" contenteditable="true">${data.general_content_title || ''}</div>
        </div>
        <div class="content-grid">
    `;
    
    if (data.contents) {
        data.contents.forEach((item, index) => {
            const contentKey = Object.keys(item).find(key => key.startsWith('content-'));
            const contentValue = contentKey ? item[contentKey] : '';
            
            contentHtml += `
                <div class="content-item">
                    <h4>${item.title || `Punkt ${String.fromCharCode(65 + index)}`}</h4>
                    <div class="editable content" 
                         data-path="presentation.content_slides.${slideIndex}.contents.${index}.${contentKey}" 
                         contenteditable="true">${contentValue}</div>
                </div>
            `;
        });
    }
    
    contentHtml += '</div>';
    return contentHtml;
}

// Create chart slide content
function createChartSlideContent(data, slideIndex) {
    let chartHtml = `
        <div>
            <label><strong>Chart Title:</strong></label>
            <div class="editable title" data-path="presentation.chart_slides.${slideIndex}.content_title" contenteditable="true">${data.content_title || ''}</div>
        </div>
        <div>
            <label><strong>Chart Summary:</strong></label>
            <div class="editable overview" data-path="presentation.chart_slides.${slideIndex}.chart_summary" contenteditable="true">${data.chart_summary || ''}</div>
        </div>
    `;
    
    if (data.chart_data && data.chart_data.length > 0) {
        const chartData = data.chart_data[0];
        const chartId = `chart-${slideIndex}-${Date.now()}`;
        
        chartHtml += `
            <div class="chart-section">
                <label><strong>Chart Preview:</strong></label>
                <div class="chart-container">
                    <canvas id="${chartId}" width="400" height="200"></canvas>
                </div>
                <div class="chart-data-editor">
                    <label><strong>Chart Data:</strong></label>
                    <div class="form-group">
                        <label>Chart Type:</label>
                        <select class="chart-type-select" data-path="presentation.chart_slides.${slideIndex}.chart_data.0.type">
                            <option value="bar" ${chartData.type === 'bar' ? 'selected' : ''}>Bar Chart</option>
                            <option value="pie" ${chartData.type === 'pie' ? 'selected' : ''}>Pie Chart</option>
                            <option value="line" ${chartData.type === 'line' ? 'selected' : ''}>Line Chart</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Chart Title:</label>
                        <input type="text" class="editable-input" data-path="presentation.chart_slides.${slideIndex}.chart_data.0.title" value="${chartData.title || ''}" />
                    </div>
        `;
        
        if (chartData.type === 'pie') {
            chartHtml += `
                    <div class="form-group">
                        <label>Labels (comma-separated):</label>
                        <input type="text" class="editable-input" data-path="presentation.chart_slides.${slideIndex}.chart_data.0.labels" value="${(chartData.labels || []).join(', ')}" />
                    </div>
                    <div class="form-group">
                        <label>Values (comma-separated):</label>
                        <input type="text" class="editable-input" data-path="presentation.chart_slides.${slideIndex}.chart_data.0.sizes" value="${(chartData.sizes || []).join(', ')}" />
                    </div>
            `;
        } else {
            chartHtml += `
                    <div class="form-group">
                        <label>X-axis Label:</label>
                        <input type="text" class="editable-input" data-path="presentation.chart_slides.${slideIndex}.chart_data.0.xlabel" value="${chartData.xlabel || ''}" />
                    </div>
                    <div class="form-group">
                        <label>Y-axis Label:</label>
                        <input type="text" class="editable-input" data-path="presentation.chart_slides.${slideIndex}.chart_data.0.ylabel" value="${chartData.ylabel || ''}" />
                    </div>
                    <div class="form-group">
                        <label>X Values (comma-separated):</label>
                        <input type="text" class="editable-input" data-path="presentation.chart_slides.${slideIndex}.chart_data.0.x" value="${(chartData.x || []).join(', ')}" />
                    </div>
                    <div class="form-group">
                        <label>Y Values (comma-separated):</label>
                        <input type="text" class="editable-input" data-path="presentation.chart_slides.${slideIndex}.chart_data.0.y" value="${(chartData.y || []).join(', ')}" />
                    </div>
            `;
        }
        
        chartHtml += `
                </div>
            </div>
        `;
        
        // Store chart info for later rendering
        setTimeout(() => {
            renderChart(chartId, chartData);
        }, 100);
    }
    
    return chartHtml;
}

// Create final slide content
function createFinalSlideContent(data) {
    let nextStepsHtml = '<div><label><strong>Next Steps:</strong></label><ul class="next-steps-list">';
    
    if (data.next_steps) {
        data.next_steps.forEach((step, index) => {
            nextStepsHtml += `
                <li>
                    <div class="editable content" 
                         data-path="presentation.final_slide.next_steps.${index}" 
                         contenteditable="true">${step}</div>
                </li>
            `;
        });
    }
    
    nextStepsHtml += '</ul></div>';
    return nextStepsHtml;
}

// Handle content editing
document.addEventListener('input', (e) => {
    if (e.target.classList.contains('editable')) {
        const path = e.target.getAttribute('data-path');
        const value = e.target.textContent;
        
        updateDataAtPath(currentData, path, value);
        
        // Debounced save to backend
        clearTimeout(window.saveTimeout);
        window.saveTimeout = setTimeout(() => {
            saveDataToBackend();
        }, 1000);
    } else if (e.target.classList.contains('editable-input')) {
        const path = e.target.getAttribute('data-path');
        let value = e.target.value;
        
        // Handle array inputs (comma-separated values)
        if (path.includes('.labels') || path.includes('.sizes') || path.includes('.x') || path.includes('.y')) {
            value = value.split(',').map(item => {
                const trimmed = item.trim();
                // Try to convert to number if it looks like a number
                const num = parseFloat(trimmed);
                return isNaN(num) ? trimmed : num;
            });
        }
        
        updateDataAtPath(currentData, path, value);
        
        // Debounced save to backend
        clearTimeout(window.saveTimeout);
        window.saveTimeout = setTimeout(() => {
            saveDataToBackend();
        }, 1000);
        
        // Re-render chart if chart data changed
        if (path.includes('chart_data')) {
            const slideIndex = path.match(/chart_slides\.(\d+)\./)[1];
            const chartId = document.querySelector(`[id^="chart-${slideIndex}-"]`).id;
            const chartData = getChartDataFromPath(currentData, slideIndex);
            renderChart(chartId, chartData);
        }
    }
});

// Handle chart type changes
document.addEventListener('change', (e) => {
    if (e.target.classList.contains('chart-type-select')) {
        const path = e.target.getAttribute('data-path');
        const value = e.target.value;
        updateDataAtPath(currentData, path, value);
        saveDataToBackend();
        
        // Re-render chart
        const slideIndex = path.match(/chart_slides\.(\d+)\./)[1];
        const chartId = document.querySelector(`[id^="chart-${slideIndex}-"]`).id;
        const chartData = getChartDataFromPath(currentData, slideIndex);
        renderChart(chartId, chartData);
    }
});

// Chart rendering function
function renderChart(canvasId, chartData) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Simple chart rendering without external libraries
    const padding = 40;
    const chartArea = {
        x: padding,
        y: padding,
        width: canvas.width - 2 * padding,
        height: canvas.height - 2 * padding
    };
    
    if (chartData.type === 'pie') {
        renderPieChart(ctx, chartArea, chartData);
    } else if (chartData.type === 'bar') {
        renderBarChart(ctx, chartArea, chartData);
    } else if (chartData.type === 'line') {
        renderLineChart(ctx, chartArea, chartData);
    }
}

// Render pie chart
function renderPieChart(ctx, chartArea, data) {
    const centerX = chartArea.x + chartArea.width / 2;
    const centerY = chartArea.y + chartArea.height / 2;
    const radius = Math.min(chartArea.width, chartArea.height) / 3;
    
    const labels = data.labels || [];
    const sizes = data.sizes || [];
    const total = sizes.reduce((sum, size) => sum + size, 0);
    
    if (total === 0) return;
    
    let currentAngle = -Math.PI / 2; // Start at top
    const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
    
    // Draw pie slices
    sizes.forEach((size, index) => {
        const sliceAngle = (size / total) * 2 * Math.PI;
        
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
        ctx.closePath();
        ctx.fillStyle = colors[index % colors.length];
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw label
        const labelAngle = currentAngle + sliceAngle / 2;
        const labelX = centerX + Math.cos(labelAngle) * (radius * 0.7);
        const labelY = centerY + Math.sin(labelAngle) * (radius * 0.7);
        
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`${labels[index] || ''}`, labelX, labelY);
        
        currentAngle += sliceAngle;
    });
}

// Render bar chart
function renderBarChart(ctx, chartArea, data) {
    const xValues = data.x || [];
    const yValues = data.y || [];
    
    if (xValues.length === 0 || yValues.length === 0) return;
    
    const maxY = Math.max(...yValues);
    const barWidth = chartArea.width / xValues.length * 0.8;
    const barSpacing = chartArea.width / xValues.length * 0.2;
    
    // Draw bars
    xValues.forEach((label, index) => {
        const value = yValues[index] || 0;
        const barHeight = (value / maxY) * chartArea.height;
        const x = chartArea.x + index * (barWidth + barSpacing) + barSpacing / 2;
        const y = chartArea.y + chartArea.height - barHeight;
        
        // Draw bar
        ctx.fillStyle = '#36A2EB';
        ctx.fillRect(x, y, barWidth, barHeight);
        
        // Draw label
        ctx.fillStyle = '#333';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(label, x + barWidth / 2, chartArea.y + chartArea.height + 15);
        
        // Draw value
        ctx.fillText(value.toString(), x + barWidth / 2, y - 5);
    });
    
    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(chartArea.x, chartArea.y + chartArea.height);
    ctx.lineTo(chartArea.x + chartArea.width, chartArea.y + chartArea.height);
    ctx.moveTo(chartArea.x, chartArea.y);
    ctx.lineTo(chartArea.x, chartArea.y + chartArea.height);
    ctx.stroke();
}

// Render line chart
function renderLineChart(ctx, chartArea, data) {
    const xValues = data.x || [];
    const yValues = data.y || [];
    
    if (xValues.length === 0 || yValues.length === 0) return;
    
    const maxY = Math.max(...yValues);
    const minY = Math.min(...yValues);
    const range = maxY - minY || 1;
    
    // Draw line
    ctx.strokeStyle = '#36A2EB';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    yValues.forEach((value, index) => {
        const x = chartArea.x + (index / (yValues.length - 1)) * chartArea.width;
        const y = chartArea.y + chartArea.height - ((value - minY) / range) * chartArea.height;
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        
        // Draw point
        ctx.fillStyle = '#36A2EB';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
    });
    
    ctx.stroke();
    
    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(chartArea.x, chartArea.y + chartArea.height);
    ctx.lineTo(chartArea.x + chartArea.width, chartArea.y + chartArea.height);
    ctx.moveTo(chartArea.x, chartArea.y);
    ctx.lineTo(chartArea.x, chartArea.y + chartArea.height);
    ctx.stroke();
}

// Helper function to get chart data from path
function getChartDataFromPath(data, slideIndex) {
    return data.presentation.chart_slides[slideIndex].chart_data[0];
}

// Update data at specific path
function updateDataAtPath(obj, path, value) {
    const parts = path.split('.');
    let current = obj;
    
    for (let i = 0; i < parts.length - 1; i++) {
        const part = parts[i];
        if (part.includes('[') && part.includes(']')) {
            // Handle array indices like "content_slides[0]"
            const [arrayName, indexStr] = part.split('[');
            const index = parseInt(indexStr.replace(']', ''));
            current = current[arrayName][index];
        } else {
            current = current[part];
        }
    }
    
    const lastPart = parts[parts.length - 1];
    if (lastPart.includes('[') && lastPart.includes(']')) {
        const [arrayName, indexStr] = lastPart.split('[');
        const index = parseInt(indexStr.replace(']', ''));
        current[arrayName][index] = value;
    } else {
        current[lastPart] = value;
    }
}

// Save data to backend
async function saveDataToBackend() {
    try {
        const response = await fetch('/update_json/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentData)
        });
        
        if (response.ok) {
            console.log('Data saved successfully');
        } else {
            throw new Error('Failed to save data');
        }
    } catch (error) {
        console.error('Error saving data:', error);
        showMessage('Error saving changes', 'error');
    }
}

// Generate PPTX
generatePptxBtn.addEventListener('click', async () => {
    setButtonLoading(generatePptxBtn, true);
    showMessage('Generating PowerPoint presentation...', 'info');
    
    try {
        const response = await fetch('/generate_pptx_from_json/', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showMessage(`Presentation generated successfully! File: ${result.pptx_file}`, 'success');
        } else {
            throw new Error(result.detail || 'Failed to generate presentation');
        }
    } catch (error) {
        console.error('Error:', error);
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        setButtonLoading(generatePptxBtn, false);
    }
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Slide Generator Frontend Loaded');
});
