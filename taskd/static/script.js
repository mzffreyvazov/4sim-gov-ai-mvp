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
        } else {
            throw new Error(result.detail || 'Failed to generate content');
        }
    } catch (error) {
        console.error('Error:', error);
        showMessage(`Error: ${error.message}`, 'error');
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
                `📊 Content Slide ${index + 1}`,
                createContentSlideContent(slide, index)
            );
            slidesContainer.appendChild(contentSlide);
        });
    }
    
    // 4. Final Slide
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
    }
});

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
