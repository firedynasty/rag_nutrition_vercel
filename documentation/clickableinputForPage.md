# Clickable Page Number Input for PDF Viewer

This document explains how to implement a clickable page number input in a React PDF viewer component, allowing users to jump directly to a specific page.

## Implementation Steps

### 1. Add State Variables

First, we added state variables to manage the editable page number:

```jsx
// Additional state for page input
const [pageInputValue, setPageInputValue] = useState("1");
const [isEditingPage, setIsEditingPage] = useState(false);
```

### 2. Create Helper Function for Direct Page Navigation

We implemented a function to handle navigation to a specific page number:

```jsx
// Go to a specific page number directly
const goToSpecificPage = useCallback((targetPage) => {
  const numericPage = parseInt(targetPage, 10);
  if (!isNaN(numericPage) && numericPage >= 1 && numericPage <= numPages) {
    setPageNumber(numericPage);
    setPageInputValue(numericPage.toString());
    // Reset scroll position
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  } else {
    // Reset input to current page if invalid
    setPageInputValue(pageNumber.toString());
  }
  // Exit editing mode
  setIsEditingPage(false);
}, [pageNumber, numPages]);
```

### 3. Update Existing Navigation Functions

We modified the existing page navigation functions to also update the page input value:

```jsx
// In changePage function
setPageNumber(newPage);
setPageInputValue(newPage.toString());

// In changePagePreserveScroll function
const newPage = pageNumber + offset;
setPageNumber(newPage);
setPageInputValue(newPage.toString());
```

### 4. Update the Document Load Handler

We ensured the page input value is set correctly when a document loads:

```jsx
const onDocumentLoadSuccess = ({ numPages }) => {
  setNumPages(numPages);
  setPageNumber(1);
  setPageInputValue("1");
  setError(null);
};
```

### 5. Create the UI Component

We replaced the static page number display with a conditional rendering that shows either:
- A clickable span when not in editing mode
- An input field when in editing mode

```jsx
<div style={{ color: '#4b5563', margin: '0 1rem', display: 'flex', alignItems: 'center' }}>
  {isEditingPage ? (
    <input
      type="text"
      value={pageInputValue}
      onChange={(e) => {
        // Allow only numbers
        const value = e.target.value.replace(/[^0-9]/g, '');
        setPageInputValue(value);
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          goToSpecificPage(pageInputValue);
        } else if (e.key === 'Escape') {
          setIsEditingPage(false);
          setPageInputValue(pageNumber.toString());
        }
      }}
      onBlur={() => {
        goToSpecificPage(pageInputValue);
      }}
      autoFocus
      style={{
        width: '3rem',
        padding: '0.25rem',
        border: '1px solid #e5e7eb',
        borderRadius: '0.25rem',
        textAlign: 'center'
      }}
      aria-label="Go to page number"
    />
  ) : (
    <span 
      onClick={() => setIsEditingPage(true)}
      style={{ 
        cursor: 'pointer',
        padding: '0.25rem 0.5rem',
        borderRadius: '0.25rem',
        backgroundColor: '#e5e7eb',
        minWidth: '1.5rem',
        textAlign: 'center'
      }}
      title="Click to edit page number"
    >
      {pageNumber}
    </span>
  )} of {numPages || '?'}
</div>
```

## Key Features

1. **Visual Indication**: The page number is styled with a gray background to indicate it's clickable
2. **Input Validation**: Only numeric input is allowed, preventing invalid entries
3. **Keyboard Navigation**: 
   - Enter key confirms the page change
   - Escape key cancels the edit
4. **Focus Management**: The input is automatically focused when clicked
5. **Error Handling**: Invalid page numbers are rejected and reset to the current page

## User Experience

1. User sees the page number displayed with a subtle background indicating it's clickable
2. Upon clicking, the number transforms into an editable input field
3. User types the desired page number
4. User confirms by pressing Enter or clicking elsewhere
5. The PDF viewer jumps directly to the specified page

This implementation provides a clean, intuitive way for users to navigate directly to specific pages without cluttering the interface with additional controls.