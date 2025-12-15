# Accessible PDF Viewer with Auto-Scroll

This React application provides an accessible PDF viewer with text layer support, auto-scroll functionality, and keyboard navigation. The app allows users to:
- Upload and view PDF files
- Navigate between pages with buttons or keyboard shortcuts
- Use auto-scroll for hands-free reading
- Zoom in and out
- Select and copy text from the PDF
- Access PDF content with screen readers and other assistive technologies

## Features

- **Sticky Navigation Bar**: Contains all controls and stays at the top while scrolling
- **Page Navigation**: Previous/Next buttons and page counter
- **Keyboard Shortcuts**:
  - 'o' - Jump to top of page (like Home key)
  - 'p' - Jump to bottom of page (like End key)
  - 'm' - Go to next page (resets scroll position)
  - ',' - Go to next page (maintains scroll position)
  - 'z' - Go to previous page
  - '/' - Toggle auto-scroll
- **Auto-scroll Functionality**: Automatically scrolls through the document
  - Adjustable scroll speed
  - Sound notifications (can be toggled on/off)
  - Smart scrolling that navigates to the next page when reaching the end
- **Text Layer Support**: The PDF viewer renders a text layer over the visual PDF, making text selectable and accessible to screen readers
- **Zoom Controls**: Increase or decrease document size
- **External Link Handling**: All links in PDF documents open in new tabs
- **Accessibility**: Built with accessibility in mind, making PDF content available to assistive technologies

## Getting Started

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

### Prerequisites

- Node.js (v14 or later recommended)
- npm or yarn

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   npm install
   ```

### Running the Application

```
npm start
```

This runs the app in development mode. Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

## Deploying to Vercel

This project is configured for easy deployment to Vercel:

1. **With Vercel CLI**:
   ```bash
   # Install Vercel CLI if you haven't already
   npm i -g vercel
   
   # Login to Vercel
   vercel login
   
   # Deploy from project directory
   vercel
   ```

2. **With GitHub Integration**:
   - Fork or push this repository to your GitHub account
   - Connect your GitHub repository to Vercel
   - Vercel will automatically detect it as a React application
   - The project will be built using the `build` script defined in package.json

## Usage

1. Load a PDF by clicking the "Choose File" button in the navigation bar
2. Navigate between pages using the Previous/Next buttons or keyboard shortcuts:
   - 'm' key for next page (resets scroll position)
   - ',' key for next page (maintains scroll position)
   - 'z' key for previous page
3. Adjust zoom level using the +/- buttons
4. Enable auto-scroll by clicking the "Auto-Scroll OFF" button (or press /)
5. Adjust auto-scroll speed using the +/- buttons next to the seconds display
6. Toggle sound notifications on/off using the radio buttons
7. Use 'o' key to jump to the top of the page and 'p' to jump to the bottom

## How It Works

The application uses:
- React for the user interface
- react-pdf/pdfjs for rendering PDFs with text layers
- Web Audio API for sound notifications
- Custom auto-scroll logic for hands-free reading
- Vercel configuration for easy deployment

## Technologies Used

- React.js
- react-pdf (wrapper for PDF.js)
- CSS for styling
- Web Audio API for sound notifications

Text in the PDF is fully accessible to screen readers and other assistive technologies. Links in PDFs will open in new tabs without disrupting your reading flow.# react-chat-report-viewer
