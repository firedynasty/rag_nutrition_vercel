# FAQ Modal Component

This document explains how the FAQ modal component works in the PDF viewer application.

## Overview

The FAQ modal is a React component that displays instructions and information about using the PDF viewer with auto-scroll functionality. It fetches content from a text file and formats it into a user-friendly, structured display.

## Features

- **Dynamic Content Loading**: Fetches FAQ content from a text file in the public directory
- **Content Formatting**: Parses plain text into structured, formatted HTML
- **Responsive Design**: Adapts to different screen sizes
- **Accessibility**: Supports keyboard navigation (Escape key to close)
- **Loading States**: Shows loading indicators and error messages when needed

## Implementation Details

### Component Structure

The `FAQModal` component (`src/components/FAQModal.js`) is a modal dialog that:

1. Takes `isOpen` and `onClose` props to control visibility and closing behavior
2. Fetches content from `/public/how_auto_scroll_works_and_key_codes.txt` when opened
3. Formats the text content into structured HTML sections
4. Renders the formatted content within a modal container

### Content Formatting

The component intelligently formats plain text into structured HTML:

1. Cleans the content of any HTML-like markup
2. Splits the content into sections (separated by multiple newlines)
3. Identifies section headings (uppercase text followed by a colon)
4. Formats each section with appropriate heading and list items
5. Highlights keyboard keys (text in single quotes) for better readability

### Styling

The modal is styled using `FAQModal.css` with features including:

- Overlay background with semi-transparency
- Centered modal dialog with maximum width/height constraints
- Responsive adjustments for mobile devices
- Styled headers, sections, and list items
- Visual states for loading and error conditions

## Usage

To use the FAQ modal in a component:

```jsx
import React, { useState } from 'react';
import FAQModal from './components/FAQModal';

function YourComponent() {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const openModal = () => setIsModalOpen(true);
  const closeModal = () => setIsModalOpen(false);

  return (
    <div>
      <button onClick={openModal}>Show Instructions</button>
      <FAQModal isOpen={isModalOpen} onClose={closeModal} />
    </div>
  );
}
```

## Content Requirements

The FAQ content file should follow this format for optimal display:

- **Section headings**: UPPERCASE followed by a colon (e.g., "KEY CONTROLS:")
- **List items**: Prefixed with "- " for bullet points
- **Keyboard keys**: Enclosed in single quotes (e.g., 'Spacebar')
- **Sections**: Separated by multiple newlines

## Customization

To customize the FAQ content:

1. Edit the text file at `/public/how_auto_scroll_works_and_key_codes.txt`
2. Follow the formatting conventions described above
3. The modal will automatically load and format the updated content

## Accessibility Considerations

- The modal can be closed with the Escape key
- Screen readers can navigate the structured content
- The modal traps focus when open (preventing interaction with background elements)

## Potential Enhancements

- Add support for markdown formatting in the source text
- Implement a search function for long FAQ documents
- Add print functionality for the instructions
- Support multiple languages or localization