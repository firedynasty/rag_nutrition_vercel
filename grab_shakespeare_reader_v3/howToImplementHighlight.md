# How to Implement Line-Based Text Highlighting System

## Overview

This is a simple, robust highlighting system that allows users to highlight entire lines/paragraphs of text with persistent storage. It was originally built for Shakespeare plays but can be adapted for any line-based content (books, scripts, poems, Bible verses, etc.).

## Key Features

- ✅ **Full-line highlighting** - Select any text in a line, highlight the entire line
- ✅ **Persistent storage** - Uses localStorage, survives page reloads
- ✅ **Context-aware** - Automatically captures metadata (speaker names, line numbers, etc.)
- ✅ **Color customization** - Change highlight colors per line
- ✅ **CSV export** - Export all highlights with metadata
- ✅ **Scene/chapter-based** - Organizes highlights by sections

## Architecture

### Core Concept

Instead of trying to track exact character positions (which breaks easily), we:
1. Let users select ANY text within a structural element (line, paragraph, verse)
2. Find which structural element contains that selection
3. Highlight the ENTIRE element
4. Store the element's index and metadata

### Storage Model

```javascript
// Storage Key Format:
// highlight_{BookTitle}_{Chapter}_{Section}

// Example:
"highlight_Romeo_and_Juliet_2_2" → [
  {
    lineIndex: 5,           // Position in the scene
    text: "ROMEO\nBut soft...",  // Full text with metadata
    speaker: "ROMEO",       // Optional metadata
    color: "#ffeb3b",       // Highlight color
    timestamp: 1761847012,  // When created
    id: "highlight-123"     // Unique identifier
  }
]
```

## Implementation Guide

### Step 1: Define Your Content Structure

Identify the HTML structure of your content. You need:
- A container element (e.g., `#output`)
- Repeating line/paragraph elements with CSS classes

**Example for Shakespeare:**
```html
<div id="output">
  <div class="line">
    <div class="speaker">ROMEO</div>
    <div class="dialogue">But soft, what light...</div>
  </div>
  <div class="stage-direction">Exit Romeo</div>
  <div class="line">...</div>
</div>
```

**Example for Bible:**
```html
<div id="output">
  <div class="verse">
    <span class="verse-number">1</span>
    <span class="verse-text">In the beginning...</span>
  </div>
  <div class="verse">...</div>
</div>
```

### Step 2: Add CSS Styles

```css
/* Highlight System Styles */
.text-highlight {
    background-color: #ffeb3b;
    padding: 2px 0;
    border-radius: 2px;
    transition: background-color 0.3s ease;
    cursor: pointer;
    position: relative;
}

.text-highlight:hover {
    filter: brightness(0.9);
}

.highlight-menu {
    position: absolute;
    background: white;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    z-index: 1001;
    min-width: 180px;
}

.highlight-menu button {
    width: 100%;
    padding: 8px 12px;
    margin: 4px 0;
    background: #f0f0f0;
    border: 1px solid #ddd;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    text-align: left;
}

.highlight-menu input[type="color"] {
    width: 100%;
    height: 35px;
    margin: 4px 0;
    cursor: pointer;
}
```

### Step 3: Core JavaScript Functions

#### 3.1 Find Line Element from Selection

```javascript
// Adapt this to match YOUR content structure
function findLineElementFromSelection(selection) {
    if (!selection.rangeCount) return null;

    const range = selection.getRangeAt(0);
    let container = range.startContainer;

    if (container.nodeType === Node.TEXT_NODE) {
        container = container.parentNode;
    }

    // Walk up to find the line element
    let lineElement = container;
    while (lineElement && lineElement.id !== 'output') {
        // CUSTOMIZE THIS: Add your line element classes
        if (lineElement.classList?.contains('line') ||
            lineElement.classList?.contains('verse') ||
            lineElement.classList?.contains('paragraph')) {
            return lineElement;
        }
        lineElement = lineElement.parentElement;
    }

    return null;
}
```

#### 3.2 Get Line Index

```javascript
function getLineIndex(lineElement) {
    const outputEl = document.getElementById('output');

    // CUSTOMIZE THIS: Match your content selectors
    const allLines = outputEl.querySelectorAll('.line, .verse, .paragraph');

    for (let i = 0; i < allLines.length; i++) {
        if (allLines[i] === lineElement) {
            return i;
        }
    }

    return -1;
}
```

#### 3.3 Storage Key Generation

```javascript
// CUSTOMIZE THIS: Match your book/chapter structure
function getHighlightStorageKey() {
    // Example variables you'd have in your app:
    // - bookTitle (e.g., "Genesis")
    // - chapterNum (e.g., 1)
    // - sectionNum (e.g., 1) - optional

    const bookTitle = currentBook.replace(/\s+/g, '_');
    const chapter = currentChapter;
    const section = currentSection || 1;

    return `highlight_${bookTitle}_${chapter}_${section}`;
}
```

#### 3.4 Highlight Creation

```javascript
function highlightSelectedText() {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    if (!selectedText) {
        alert('Please select some text first.');
        return;
    }

    const lineElement = findLineElementFromSelection(selection);
    if (!lineElement) {
        alert('Please select text from the content.');
        return;
    }

    const lineIndex = getLineIndex(lineElement);
    if (lineIndex === -1) {
        alert('Could not determine line position.');
        return;
    }

    // CUSTOMIZE THIS: Extract metadata from your line structure
    let fullText = '';
    let metadata = {};

    // Example for dialogue:
    const speakerEl = lineElement.querySelector('.speaker');
    const dialogueEl = lineElement.querySelector('.dialogue');
    if (speakerEl) {
        metadata.speaker = speakerEl.textContent.trim();
        fullText = metadata.speaker + '\n';
    }
    if (dialogueEl) {
        fullText += dialogueEl.textContent.trim();
    }

    // Example for verses:
    const verseNum = lineElement.querySelector('.verse-number');
    const verseText = lineElement.querySelector('.verse-text');
    if (verseNum) {
        metadata.verseNumber = verseNum.textContent.trim();
        fullText = `[${metadata.verseNumber}] `;
    }
    if (verseText) {
        fullText += verseText.textContent.trim();
    }

    const highlightData = {
        lineIndex: lineIndex,
        text: fullText,
        metadata: metadata,  // Store any extra data
        timestamp: Date.now(),
        color: '#ffeb3b'
    };

    const highlightId = saveHighlight(highlightData);
    if (highlightId) {
        highlightData.id = highlightId;
        applyHighlightToLine(lineElement, highlightData);
        selection.removeAllRanges();

        // Show success feedback
        alert('✅ Highlighted!');
    }
}
```

#### 3.5 Apply Visual Highlight

```javascript
function applyHighlightToLine(lineElement, highlightData) {
    if (!lineElement) return;

    // Add highlight styling
    lineElement.classList.add('text-highlight');
    lineElement.style.backgroundColor = highlightData.color;
    lineElement.dataset.lineIndex = highlightData.lineIndex;
    lineElement.style.cursor = 'pointer';
    lineElement.style.borderRadius = '4px';
    lineElement.style.padding = '8px';
    lineElement.style.margin = '8px 0';

    // Add click handler for menu
    lineElement.onclick = (e) => {
        e.stopPropagation();
        showHighlightMenu(lineElement, highlightData);
    };
}
```

#### 3.6 LocalStorage Management

```javascript
// Save highlight
function saveHighlight(highlightData) {
    const key = getHighlightStorageKey();
    if (!key) return;

    const existing = JSON.parse(localStorage.getItem(key) || '[]');

    // Check if line already highlighted
    const existingIndex = existing.findIndex(h => h.lineIndex === highlightData.lineIndex);
    if (existingIndex >= 0) {
        // Replace existing
        highlightData.id = existing[existingIndex].id;
        existing[existingIndex] = highlightData;
    } else {
        // Add new
        highlightData.id = 'highlight-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        existing.push(highlightData);
    }

    localStorage.setItem(key, JSON.stringify(existing));
    return highlightData.id;
}

// Delete highlight
function deleteHighlight(lineIndex) {
    const key = getHighlightStorageKey();
    if (!key) return;

    const existing = JSON.parse(localStorage.getItem(key) || '[]');
    const filtered = existing.filter(h => h.lineIndex !== lineIndex);

    localStorage.setItem(key, JSON.stringify(filtered));
}

// Update color
function updateHighlightColor(lineIndex, color) {
    const key = getHighlightStorageKey();
    if (!key) return;

    const existing = JSON.parse(localStorage.getItem(key) || '[]');
    const highlight = existing.find(h => h.lineIndex === lineIndex);

    if (highlight) {
        highlight.color = color;
        localStorage.setItem(key, JSON.stringify(existing));
    }
}
```

#### 3.7 Restore Highlights

```javascript
// Call this after rendering content
function restoreHighlights() {
    const key = getHighlightStorageKey();
    if (!key) return;

    const highlightsJSON = localStorage.getItem(key);
    if (!highlightsJSON) return;

    try {
        const highlights = JSON.parse(highlightsJSON);

        // Get all line elements
        const outputEl = document.getElementById('output');
        const allLines = outputEl.querySelectorAll('.line, .verse, .paragraph');

        highlights.forEach(highlightData => {
            const lineElement = allLines[highlightData.lineIndex];
            if (lineElement) {
                applyHighlightToLine(lineElement, highlightData);
            }
        });
    } catch (error) {
        console.error('Failed to restore highlights:', error);
    }
}
```

#### 3.8 Context Menu

```javascript
function showHighlightMenu(element, highlightData) {
    // Remove any existing menu
    const existingMenu = document.querySelector('.highlight-menu');
    if (existingMenu) existingMenu.remove();

    const menu = document.createElement('div');
    menu.className = 'highlight-menu';
    const rect = element.getBoundingClientRect();
    menu.style.position = 'absolute';
    menu.style.top = `${rect.top + window.scrollY - 60}px`;
    menu.style.left = `${rect.left}px`;

    menu.innerHTML = `
        <button class="close-btn" style="float: right;">✕</button>
        <div style="clear: both; padding-top: 8px;">
            <button class="delete-highlight">🗑️ Delete Highlight</button>
            <label style="display: block; font-size: 11px; margin: 8px 0 4px 0;">Change Color:</label>
            <input type="color" class="change-color" value="${highlightData.color}">
        </div>
    `;

    menu.querySelector('.close-btn').onclick = () => menu.remove();

    menu.querySelector('.delete-highlight').onclick = () => {
        const lineIndex = parseInt(element.dataset.lineIndex);
        deleteHighlight(lineIndex);

        // Remove styling
        element.classList.remove('text-highlight');
        element.style.backgroundColor = '';
        element.style.cursor = '';
        element.style.borderRadius = '';
        element.style.padding = '';
        element.style.margin = '';
        element.onclick = null;
        delete element.dataset.lineIndex;

        menu.remove();
    };

    menu.querySelector('.change-color').onchange = (e) => {
        const lineIndex = parseInt(element.dataset.lineIndex);
        element.style.backgroundColor = e.target.value;
        updateHighlightColor(lineIndex, e.target.value);
    };

    document.body.appendChild(menu);

    setTimeout(() => {
        document.addEventListener('click', () => menu.remove(), { once: true });
    }, 10);
}
```

### Step 4: Export Functionality

```javascript
function exportHighlights() {
    const allHighlights = [];

    // Iterate through all localStorage keys
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith('highlight_')) {
            const highlightsJSON = localStorage.getItem(key);
            if (highlightsJSON) {
                try {
                    const highlights = JSON.parse(highlightsJSON);

                    // Parse key: highlight_BookTitle_Chapter_Section
                    const parts = key.replace('highlight_', '').split('_');
                    const section = parts.pop();
                    const chapter = parts.pop();
                    const bookTitle = parts.join(' ');

                    highlights.forEach(h => {
                        allHighlights.push({
                            title: bookTitle,
                            chapter: chapter,
                            section: section,
                            // CUSTOMIZE: Add your metadata fields
                            speaker: h.metadata?.speaker || '',
                            verseNumber: h.metadata?.verseNumber || '',
                            highlight: h.text,
                            color: h.color,
                            timestamp: h.timestamp
                        });
                    });
                } catch (e) {
                    console.error('Error parsing highlights:', e);
                }
            }
        }
    }

    if (allHighlights.length === 0) {
        alert('No highlights to export.');
        return;
    }

    // Sort by timestamp
    allHighlights.sort((a, b) => a.timestamp - b.timestamp);

    // Create CSV
    // CUSTOMIZE: Add your columns
    let csv = 'Title,Chapter,Section,Speaker,Highlight\n';
    allHighlights.forEach(h => {
        const highlight = h.highlight.replace(/"/g, '""');
        const needsQuotes = highlight.includes(',') || highlight.includes('\n');
        const formattedHighlight = needsQuotes ? `"${highlight}"` : highlight;

        csv += `${h.title},${h.chapter},${h.section},${h.speaker},${formattedHighlight}\n`;
    });

    // Copy to clipboard
    if (navigator.clipboard) {
        navigator.clipboard.writeText(csv).then(() => {
            alert(`Exported ${allHighlights.length} highlights to clipboard!`);
        }).catch(err => {
            console.error('Clipboard error:', err);
            // Fallback: show in modal
            showExportInModal(csv);
        });
    } else {
        showExportInModal(csv);
    }
}

function showExportInModal(csv) {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        z-index: 10000;
        max-width: 80%;
    `;

    modal.innerHTML = `
        <h3>Exported Highlights</h3>
        <p>Copy the text below:</p>
        <textarea readonly style="width: 100%; height: 300px; font-family: monospace;">${csv}</textarea>
        <button onclick="this.parentElement.remove()">Close</button>
    `;

    document.body.appendChild(modal);
    modal.querySelector('textarea').select();
}
```

### Step 5: UI Integration

Add buttons to your interface:

```html
<button onclick="highlightSelectedText()">🖍️ Highlight</button>
<button onclick="removeAllHighlights()">🗑️ Clear All</button>
<button onclick="exportHighlights()">📥 Export</button>
```

### Step 6: Initialize

```javascript
// When content is loaded/changed, restore highlights
function displayContent() {
    // Your code to render content
    renderChapter();

    // Restore highlights after content is rendered
    restoreHighlights();
}

// On page load
document.addEventListener('DOMContentLoaded', function() {
    displayContent();
});
```

## Customization Examples

### For Bible Verses

```javascript
// Storage key
function getHighlightStorageKey() {
    return `highlight_${currentBook}_${currentChapter}_1`;
}

// Extract metadata
const verseNum = lineElement.querySelector('.verse-number').textContent;
const verseText = lineElement.querySelector('.verse-text').textContent;
const highlightData = {
    lineIndex: lineIndex,
    text: `[${verseNum}] ${verseText}`,
    metadata: { verseNumber: verseNum },
    timestamp: Date.now(),
    color: '#ffeb3b'
};
```

### For Novel Paragraphs

```javascript
// Storage key
function getHighlightStorageKey() {
    return `highlight_${bookTitle}_chapter_${chapterNum}`;
}

// Extract metadata
const paragraphText = lineElement.textContent.trim();
const highlightData = {
    lineIndex: lineIndex,
    text: paragraphText,
    metadata: {
        chapterTitle: currentChapterTitle,
        pageNumber: getCurrentPage()
    },
    timestamp: Date.now(),
    color: '#ffeb3b'
};
```

### For Poetry

```javascript
// Storage key
function getHighlightStorageKey() {
    return `highlight_${poemTitle}_stanza_${stanzaNum}`;
}

// Extract metadata
const lineNum = lineElement.dataset.lineNumber;
const lineText = lineElement.textContent.trim();
const highlightData = {
    lineIndex: lineIndex,
    text: lineText,
    metadata: {
        lineNumber: lineNum,
        stanzaNumber: currentStanza
    },
    timestamp: Date.now(),
    color: '#ffeb3b'
};
```

## Troubleshooting

### Highlights don't appear after reload

**Problem:** `restoreHighlights()` not being called after content renders.

**Solution:** Make sure to call `restoreHighlights()` AFTER the DOM is updated:

```javascript
function displayChapter() {
    outputEl.innerHTML = renderedHTML;
    restoreHighlights();  // Call here!
}
```

### Wrong lines get highlighted

**Problem:** Line indices change when content structure changes.

**Solution:** Highlights are tied to position. If you insert/remove lines, indices shift. Consider adding a unique identifier to each line if needed:

```html
<div class="line" data-line-id="act1-scene2-line5">
```

Then store `lineId` instead of `lineIndex`.

### LocalStorage quota exceeded

**Problem:** Too many highlights stored.

**Solution:** LocalStorage has ~5-10MB limit. For large-scale usage, consider:
- IndexedDB instead of localStorage
- Compression of stored data
- Periodic cleanup of old highlights

### Highlights break on dynamic font sizing

**Problem:** CSS `font-size` changes via JavaScript don't get restored.

**Solution:** Reapply font sizes after restoring highlights:

```javascript
function restoreHighlights() {
    // ... restore highlights ...
    applyFontSizes();  // Reapply any custom font sizing
}
```

## Performance Tips

1. **Debounce highlight restoration** if rendering frequently:
```javascript
let restoreTimeout;
function scheduleHighlightRestore() {
    clearTimeout(restoreTimeout);
    restoreTimeout = setTimeout(restoreHighlights, 100);
}
```

2. **Batch localStorage operations** when deleting multiple highlights:
```javascript
function removeAllHighlights() {
    const key = getHighlightStorageKey();
    localStorage.removeItem(key);  // Single operation
    // Then update UI
}
```

3. **Use event delegation** for highlight clicks instead of individual handlers:
```javascript
document.getElementById('output').addEventListener('click', (e) => {
    const highlightedElement = e.target.closest('.text-highlight');
    if (highlightedElement) {
        const lineIndex = parseInt(highlightedElement.dataset.lineIndex);
        // Get highlight data and show menu
    }
});
```

## Security Considerations

- **XSS Prevention:** Always sanitize text before displaying in alerts/modals
- **Storage Limits:** Validate data size before storing
- **Privacy:** Consider adding an option to export/clear all data

## Future Enhancements

Possible additions to this system:

1. **Notes/Annotations:** Add a text field to store notes with each highlight
2. **Categories/Tags:** Allow users to categorize highlights
3. **Search:** Search within highlighted text
4. **Sync:** Sync highlights across devices via backend API
5. **Sharing:** Generate shareable links to highlighted passages
6. **Import:** Import highlights from CSV
7. **Multiple Colors:** Preset color schemes for different purposes

## License

This code is provided as-is for educational purposes. Feel free to adapt for your projects.

---

**Questions?** Review the implementation in `indexRegular.html` lines 1265-1730 for the complete working example.
