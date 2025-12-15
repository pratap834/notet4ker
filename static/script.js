/**
 * Physician Notetaker - Frontend JavaScript
 * Handles user interactions and API communication
 */

// Sample transcript for demo purposes
const SAMPLE_TRANSCRIPT = `Doctor: Good morning, how are you feeling today?
Patient: I've been experiencing severe headaches for the past week, especially in the mornings.
Doctor: I see. Any other symptoms like nausea or sensitivity to light?
Patient: Yes, I feel nauseous sometimes and bright lights make it worse. It's really affecting my work.
Doctor: How long do these headaches typically last?
Patient: Usually a few hours, but sometimes they last all day.
Doctor: Have you taken any medication for this?
Patient: I tried some over-the-counter painkillers but they don't help much.
Doctor: Based on your symptoms - severe headaches, nausea, and photophobia - this appears to be migraines. I'm prescribing sumatriptan 50mg to take at the onset of symptoms.
Patient: How long will it take to feel better?
Doctor: The medication should start working within 30 minutes to 2 hours. You should see significant improvement within a few days of consistent treatment.
Patient: Are there any lifestyle changes I should make?
Doctor: Yes, try to maintain a regular sleep schedule, stay hydrated, avoid known triggers like caffeine and stress, and keep a headache diary to identify patterns.
Patient: Okay, thank you doctor.
Doctor: You're welcome. Let's schedule a follow-up in two weeks to assess how the treatment is working. If symptoms worsen or you experience any side effects, please contact us immediately.`;

// Global variable to store results
let currentResults = null;

// DOM Elements
const transcriptInput = document.getElementById('transcript-input');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const loadSampleBtn = document.getElementById('load-sample-btn');
const exportBtn = document.getElementById('export-btn');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('error-message');
const resultsSection = document.getElementById('results-section');

/**
 * Toggle collapsible sections
 */
function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    const header = section.previousElementSibling;
    const icon = header.querySelector('.toggle-icon');
    
    if (section.style.display === 'none') {
        section.style.display = 'block';
        icon.textContent = '▼';
        header.classList.remove('collapsed');
    } else {
        section.style.display = 'none';
        icon.textContent = '▶';
        header.classList.add('collapsed');
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    analyzeBtn.addEventListener('click', analyzeTranscript);
    clearBtn.addEventListener('click', clearAll);
    loadSampleBtn.addEventListener('click', loadSample);
    exportBtn.addEventListener('click', exportResults);
    
    // Allow Ctrl+Enter to submit
    transcriptInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeTranscript();
        }
    });
});

/**
 * Load sample transcript into the input
 */
function loadSample() {
    transcriptInput.value = SAMPLE_TRANSCRIPT;
    transcriptInput.focus();
}

/**
 * Clear all inputs and results
 */
function clearAll() {
    transcriptInput.value = '';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    currentResults = null;
    transcriptInput.focus();
}

/**
 * Main function to analyze transcript
 */
async function analyzeTranscript() {
    const transcript = transcriptInput.value.trim();
    
    // Validation
    if (!transcript) {
        showError('Please enter a medical transcript to analyze.');
        return;
    }
    
    if (transcript.length < 10) {
        showError('Transcript is too short. Please provide at least 10 characters.');
        return;
    }
    
    // Show loading state
    setLoadingState(true);
    hideError();
    resultsSection.style.display = 'none';
    
    try {
        // Call the API
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ transcript: transcript })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to analyze transcript');
        }
        
        const data = await response.json();
        currentResults = data;
        
        // Display results
        displayResults(data);
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
        
    } catch (error) {
        console.error('Error:', error);
        showError(`Error analyzing transcript: ${error.message}`);
    } finally {
        setLoadingState(false);
    }
}

/**
 * Display the analysis results
 */
function displayResults(data) {
    // Display SOAP Note
    displaySOAPNote(data.soap_note);
    
    // Display Summary
    displaySummary(data.summary);
    
    // Display Entities
    displayEntities(data.entities);
    
    // Display Metadata
    displayMetadata(data);
}

/**
 * Display SOAP Note
 */
function displaySOAPNote(soapNote) {
    const soapContainer = document.getElementById('soap-note');
    soapContainer.innerHTML = '';
    
    const sections = ['Subjective', 'Objective', 'Assessment', 'Plan'];
    
    sections.forEach(section => {
        if (soapNote[section]) {
            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'soap-section';
            
            const heading = document.createElement('h4');
            heading.textContent = section.toUpperCase();
            
            const content = document.createElement('p');
            content.textContent = soapNote[section].content || soapNote[section];
            
            sectionDiv.appendChild(heading);
            sectionDiv.appendChild(content);
            soapContainer.appendChild(sectionDiv);
        }
    });
}

/**
 * Display Clinical Summary
 */
function displaySummary(summary) {
    const summaryContainer = document.getElementById('summary');
    summaryContainer.innerHTML = '';
    
    const paragraph = document.createElement('p');
    paragraph.textContent = summary;
    summaryContainer.appendChild(paragraph);
}

/**
 * Display Extracted Entities
 */
function displayEntities(entities) {
    // Symptoms
    displayEntityList('symptoms-list', entities.Symptoms || []);
    
    // Diagnosis
    displayEntityList('diagnosis-list', entities.Diagnosis || []);
    
    // Treatment
    displayEntityList('treatment-list', entities.Treatment || []);
    
    // Medications
    displayEntityList('medications-list', entities.Medications || []);
}

/**
 * Display a list of entities
 */
function displayEntityList(elementId, items) {
    const listElement = document.getElementById(elementId);
    listElement.innerHTML = '';
    
    if (items.length === 0) {
        listElement.className = 'entity-list empty';
        listElement.textContent = 'None identified';
        return;
    }
    
    listElement.className = 'entity-list';
    
    // Remove duplicates and create list items
    const uniqueItems = [...new Set(items)];
    
    uniqueItems.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        listElement.appendChild(li);
    });
}

/**
 * Display metadata and analysis details
 */
function displayMetadata(data) {
    const metadataContainer = document.getElementById('metadata');
    metadataContainer.innerHTML = '';
    
    const metadata = [
        {
            label: 'Sentiment',
            value: `${data.sentiment.sentiment} (${(data.sentiment.confidence * 100).toFixed(1)}% confidence)`
        },
        {
            label: 'Intent',
            value: `${data.intent.intent} (${(data.intent.confidence * 100).toFixed(1)}% confidence)`
        },
        {
            label: 'Processing Device',
            value: data.model_info.device
        },
        {
            label: 'NER Model',
            value: data.model_info.ner
        },
        {
            label: 'Sentiment Model',
            value: data.model_info.sentiment
        },
        {
            label: 'Summarization Model',
            value: data.model_info.summarization
        }
    ];
    
    metadata.forEach(item => {
        const div = document.createElement('div');
        div.className = 'metadata-item';
        
        const label = document.createElement('strong');
        label.textContent = item.label;
        
        const value = document.createElement('span');
        value.textContent = item.value;
        
        div.appendChild(label);
        div.appendChild(value);
        metadataContainer.appendChild(div);
    });
}

/**
 * Export results as JSON
 */
function exportResults() {
    if (!currentResults) {
        showError('No results to export. Please analyze a transcript first.');
        return;
    }
    
    // Create a blob with the JSON data
    const dataStr = JSON.stringify(currentResults, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    
    // Create download link
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `physician_notetaker_results_${Date.now()}.json`;
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    
    // Cleanup
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

/**
 * Show error message
 */
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.style.display = 'none';
}

/**
 * Set loading state
 */
function setLoadingState(isLoading) {
    if (isLoading) {
        loading.style.display = 'block';
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
    } else {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 6v6l4 2"/>
            </svg>
            Analyze Transcript
        `;
    }
}
