import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import HardcodedFAQModal from './components/HardcodedFAQModal';

// Set up the worker for PDF.js
// This is required for react-pdf to work
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

const AccessiblePDFViewer = () => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [pageInputValue, setPageInputValue] = useState("1");
  const [isEditingPage, setIsEditingPage] = useState(false);
  const [scale, setScale] = useState(1.0);
  const [pdfFile, setPdfFile] = useState(null);
  const [error, setError] = useState(null);
  
  // Auto-scroll state
  const [autoScrollActive, setAutoScrollActive] = useState(false);
  const [scrollIntervalSeconds, setScrollIntervalSeconds] = useState(10);
  const [soundEnabled, setSoundEnabled] = useState(true);
  
  // FAQ modal state
  const [isFAQModalOpen, setIsFAQModalOpen] = useState(false);
  
  // Refs for auto-scroll
  const lastScrollTimeRef = useRef(Date.now());
  const audioContextRef = useRef(null);

  // Add text layer styles to document
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .react-pdf__Page__textContent {
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        overflow: hidden;
        line-height: 1.0;
        position: absolute;
      }
      .react-pdf__Page__textContent > span {
        color: transparent;
        position: absolute;
        white-space: pre;
        cursor: text;
        transform-origin: 0% 0%;
      }
      .react-pdf__Page__textContent > span::selection {
        background-color: rgba(0, 128, 255, 0.3);
      }
      .react-pdf__Page__annotations {
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        overflow: hidden;
        pointer-events: none;
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, []);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setPageNumber(1);
    setPageInputValue("1");
    setError(null);
  };

  const onDocumentLoadError = (error) => {
    console.error('Error loading PDF:', error);
    setError('Failed to load PDF. Please try a different file.');
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      setPdfFile(file);
      setError(null);
    } else if (file) {
      setError('Please select a valid PDF file.');
      setPdfFile(null);
    }
  };

  // Basic page navigation - resets scroll position
  const changePage = useCallback((offset) => {
    const newPage = pageNumber + offset;
    if (newPage >= 1 && newPage <= numPages) {
      setPageNumber(newPage);
      setPageInputValue(newPage.toString());
      // Reset scroll position when changing pages
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    }
  }, [pageNumber, numPages]);
  
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

  // Change page while preserving scroll position
  const changePagePreserveScroll = useCallback((offset) => {
    if (pageNumber + offset >= 1 && pageNumber + offset <= numPages) {
      // Store current scroll position
      const currentScrollPosition = window.scrollY;
      
      // Update page number
      const newPage = pageNumber + offset;
      setPageNumber(newPage);
      setPageInputValue(newPage.toString());
      
      // Restore scroll position after a short delay to allow page rendering
      setTimeout(() => {
        window.scrollTo({
          top: currentScrollPosition,
          behavior: 'auto'
        });
      }, 100);
    }
  }, [pageNumber, numPages]);

  const changeZoom = useCallback((newScale) => {
    setScale(newScale);
  }, []);
  
  // Play a subtle beep sound for auto-scroll notifications
  const playSubtleBeep = useCallback(() => {
    if (!soundEnabled) return;
    
    try {
      // Create audio context on first use
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      
      const context = audioContextRef.current;
      
      // Create oscillator for sound generation
      const oscillator = context.createOscillator();
      const gainNode = context.createGain();
      
      // Configure sound properties (gentle, quiet beep)
      oscillator.type = 'sine';
      oscillator.frequency.value = 800;
      gainNode.gain.value = 0.05;
      
      // Connect audio nodes
      oscillator.connect(gainNode);
      gainNode.connect(context.destination);
      
      // Play very short beep
      gainNode.gain.setValueAtTime(0.05, context.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.15);
      oscillator.start();
      oscillator.stop(context.currentTime + 0.15);
    } catch (error) {
      console.error('Error playing notification sound:', error);
    }
  }, [soundEnabled]);
  
  // Reset the auto-scroll timer
  const resetScrollTimer = useCallback(() => {
    lastScrollTimeRef.current = Date.now();
  }, []);

  // Toggle auto-scroll on/off
  const toggleAutoScroll = useCallback(() => {
    setAutoScrollActive(prev => {
      // Reset timer when turning on
      if (!prev) {
        lastScrollTimeRef.current = Date.now();
      }
      return !prev;
    });
  }, []);
  
  // Auto-scroll speed controls
  const increaseSpeed = () => {
    setScrollIntervalSeconds(prev => Math.max(1, prev - 1));
  };
  
  const decreaseSpeed = () => {
    setScrollIntervalSeconds(prev => Math.min(60, prev + 1));
  };
  
  // Track window position
  const [windowPosition, setWindowPosition] = useState('top');
  
  // Perform auto-scroll action
  const performScroll = useCallback(() => {
    try {
      if (windowPosition === 'top' || windowPosition === 'middle') {
        // If we're at the top or middle, scroll to the bottom
        window.scrollTo({
          top: document.documentElement.scrollHeight,
          behavior: 'smooth'
        });
        
        setWindowPosition('bottom');
        
        // Play sound if enabled
        if (soundEnabled) {
          playSubtleBeep();
        }
      } else if (windowPosition === 'bottom') {
        // We're at the bottom, go to next page
        if (pageNumber < numPages) {
          // Go to next page
          changePagePreserveScroll(1);
          
          // Reset position to top for new page
          setTimeout(() => {
            window.scrollTo({
              top: 0,
              behavior: 'auto'
            });
            setWindowPosition('top');
          }, 100);
          
          // Play sound if enabled
          if (soundEnabled) {
            playSubtleBeep();
          }
        } else {
          // We're at the last page, go back to top
          window.scrollTo({
            top: 0,
            behavior: 'smooth'
          });
          setWindowPosition('top');
          
          // Play sound if enabled
          if (soundEnabled) {
            playSubtleBeep();
          }
        }
      }
      
      // Reset timer after action
      resetScrollTimer();
    } catch (error) {
      console.error('Error during auto-scroll:', error);
      resetScrollTimer();
    }
  }, [numPages, pageNumber, playSubtleBeep, resetScrollTimer, soundEnabled, changePagePreserveScroll, windowPosition]);
  
  // Update window position on scroll
  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;
      
      // Determine position
      if (scrollY < 20) {
        setWindowPosition('top');
      } else if (Math.abs((scrollY + windowHeight) - documentHeight) < 20) {
        setWindowPosition('bottom');
      } else {
        setWindowPosition('middle');
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  // Auto-scroll interval effect
  useEffect(() => {
    let intervalId;
    let errorCount = 0;
    
    if (autoScrollActive) {
      intervalId = setInterval(() => {
        try {
          // Get current time
          const now = Date.now();
          // Only scroll if enough time has passed
          if (now - lastScrollTimeRef.current >= scrollIntervalSeconds * 1000) {
            performScroll();
            // Reset error count on successful scroll
            if (errorCount > 0) errorCount = 0;
          }
        } catch (error) {
          // Error recovery strategy
          errorCount++;
          console.error(`Auto-scroll error (${errorCount}/3):`, error);
          
          // If we encounter too many errors, disable auto-scroll
          if (errorCount >= 3) {
            console.warn("Too many auto-scroll errors, disabling auto-scroll");
            setAutoScrollActive(false);
            errorCount = 0;
            
            // Show error message to user
            alert("Auto-scroll has been disabled due to errors. You can try enabling it again.");
          }
          
          // Update last scroll time to prevent rapid retries
          lastScrollTimeRef.current = Date.now();
        }
      }, 1000); // Check every second
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoScrollActive, performScroll, scrollIntervalSeconds]);

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      switch(e.key) {
        case 'o':
        case 'O':
          // Scroll to top of the page (like Home key)
          window.scrollTo({
            top: 0,
            behavior: 'smooth'
          });
          resetScrollTimer();
          e.preventDefault();
          break;
        case 'p':
        case 'P':
          // Scroll to bottom of the page (like End key)
          window.scrollTo({
            top: document.body.scrollHeight,
            behavior: 'smooth'
          });
          resetScrollTimer();
          e.preventDefault();
          break;
        case 'm':
        case 'M':
          // Go to next page (resets scroll position)
          changePage(1);
          resetScrollTimer();
          e.preventDefault();
          break;
        case ',':
          // Go to next page but maintain scroll position
          changePagePreserveScroll(1);
          resetScrollTimer();
          e.preventDefault();
          break;
        case 'z':
        case 'Z':
          // Go to previous page
          changePage(-1);
          resetScrollTimer();
          e.preventDefault();
          break;
        case '/':
          // Toggle auto-scroll
          toggleAutoScroll();
          e.preventDefault();
          break;
        default:
          break;
      }
    };
    
    // Only add event listener if PDF is loaded
    if (pdfFile) {
      document.addEventListener('keydown', handleKeyDown);
      
      // Add handler for PDF links to open in new tabs
      const handleLinkClick = (e) => {
        // Check if the click is on a link within the PDF
        let target = e.target;
        
        // Look for parent links if the target itself is not a link
        while (target && target.tagName !== 'A' && target.parentNode) {
          target = target.parentNode;
        }
        
        // If we found a link within the PDF viewer
        if (target && target.tagName === 'A' && 
            (target.closest('.react-pdf__Page') || target.closest('.react-pdf__Page__annotations'))) {
          e.preventDefault();
          
          // Get the href
          const href = target.getAttribute('href');
          
          if (href && !href.startsWith('#')) {
            // Open in new tab
            window.open(href, '_blank', 'noopener,noreferrer');
          }
        }
      };
      
      // Add the click listener to the document
      document.addEventListener('click', handleLinkClick, true);
      
      return () => {
        document.removeEventListener('keydown', handleKeyDown);
        document.removeEventListener('click', handleLinkClick, true);
      };
    }
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [pdfFile, changePage, changePagePreserveScroll, resetScrollTimer, toggleAutoScroll]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '1rem', width: '100%' }}>
      {/* Sticky Navbar */}
      {pdfFile && (
        <div style={{
          position: 'sticky',
          top: 0,
          width: '100%',
          maxWidth: '64rem',
          backgroundColor: '#ffffff',
          borderBottom: '1px solid #e5e7eb',
          padding: '0.75rem 1rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
          zIndex: 1000,
          marginBottom: '1rem',
          borderRadius: '0.5rem'
        }}>
          {/* Navigation Controls - Left */}
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <button 
              onClick={() => changePage(-1)} 
              disabled={pageNumber <= 1}
              style={{ 
                padding: '0.5rem 1rem', 
                backgroundColor: pageNumber <= 1 ? '#9ca3af' : '#2563eb', 
                color: 'white', 
                borderRadius: '0.375rem',
                marginRight: '0.5rem',
                border: 'none',
                cursor: pageNumber <= 1 ? 'not-allowed' : 'pointer'
              }}
            >
              Previous (z)
            </button>
            <button 
              onClick={() => changePage(1)} 
              disabled={pageNumber >= numPages}
              style={{ 
                padding: '0.5rem 1rem', 
                backgroundColor: pageNumber >= numPages ? '#9ca3af' : '#2563eb', 
                color: 'white', 
                borderRadius: '0.375rem',
                border: 'none',
                cursor: pageNumber >= numPages ? 'not-allowed' : 'pointer'
              }}
            >
              Next (m/,)
            </button>
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
          </div>

          {/* Zoom Controls - Right */}
          <div style={{ display: 'flex', alignItems: 'center' }}>
            {/* Zoom controls */}
            <div style={{ display: 'flex', alignItems: 'center', marginRight: '1.5rem' }}>
              <button 
                onClick={() => changeZoom(Math.max(0.5, scale - 0.2))} 
                style={{ 
                  padding: '0.25rem 0.75rem', 
                  backgroundColor: '#e5e7eb', 
                  borderRadius: '0.375rem',
                  marginRight: '0.5rem',
                  border: 'none',
                  cursor: 'pointer'
                }}
              >
                -
              </button>
              <span style={{ color: '#4b5563' }}>{Math.round(scale * 100)}%</span>
              <button 
                onClick={() => changeZoom(scale + 0.2)}
                style={{ 
                  padding: '0.25rem 0.75rem', 
                  backgroundColor: '#e5e7eb', 
                  borderRadius: '0.375rem',
                  marginLeft: '0.5rem',
                  border: 'none',
                  cursor: 'pointer'
                }}
              >
                +
              </button>
            </div>
            
            {/* Auto-scroll controls */}
            <div style={{ display: 'flex', alignItems: 'center', marginRight: '1.5rem', borderLeft: '1px solid #e5e7eb', paddingLeft: '1rem' }}>
              <button 
                onClick={toggleAutoScroll}
                style={{ 
                  padding: '0.25rem 0.75rem', 
                  backgroundColor: autoScrollActive ? '#059669' : '#e5e7eb',
                  color: autoScrollActive ? 'white' : '#4b5563',
                  borderRadius: '0.375rem',
                  marginRight: '1rem',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '0.875rem',
                  minWidth: '145px',
                  whiteSpace: 'nowrap'
                }}
              >
                Auto-Scroll {autoScrollActive ? 'ON' : 'OFF'} (/)
              </button>
              
              <div style={{ display: 'flex', alignItems: 'center', marginRight: '1rem' }}>
                <button 
                  onClick={increaseSpeed} 
                  disabled={scrollIntervalSeconds <= 1}
                  style={{ 
                    padding: '0.25rem 0.5rem', 
                    backgroundColor: scrollIntervalSeconds <= 1 ? '#9ca3af' : '#e5e7eb',
                    borderRadius: '0.375rem',
                    marginRight: '0.5rem',
                    border: 'none',
                    cursor: scrollIntervalSeconds <= 1 ? 'not-allowed' : 'pointer'
                  }}
                >
                  -
                </button>
                <span style={{ color: '#4b5563', minWidth: '2.5rem', textAlign: 'center' }}>{scrollIntervalSeconds}s</span>
                <button 
                  onClick={decreaseSpeed} 
                  disabled={scrollIntervalSeconds >= 60}
                  style={{ 
                    padding: '0.25rem 0.5rem', 
                    backgroundColor: scrollIntervalSeconds >= 60 ? '#9ca3af' : '#e5e7eb',
                    borderRadius: '0.375rem',
                    marginLeft: '0.5rem',
                    border: 'none',
                    cursor: scrollIntervalSeconds >= 60 ? 'not-allowed' : 'pointer'
                  }}
                >
                  +
                </button>
              </div>
              
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <span style={{ marginRight: '0.5rem', fontSize: '0.875rem', color: '#4b5563' }}>Sound:</span>
                <label style={{ display: 'flex', alignItems: 'center', marginRight: '0.5rem', cursor: 'pointer' }}>
                  <input
                    type="radio"
                    checked={soundEnabled}
                    onChange={() => setSoundEnabled(true)}
                    style={{ marginRight: '0.25rem' }}
                  />
                  <span style={{ fontSize: '0.875rem', color: '#4b5563' }}>On</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                  <input
                    type="radio"
                    checked={!soundEnabled}
                    onChange={() => setSoundEnabled(false)}
                    style={{ marginRight: '0.25rem' }}
                  />
                  <span style={{ fontSize: '0.875rem', color: '#4b5563' }}>Off</span>
                </label>
              </div>
            </div>
            
            {/* FAQ Button */}
            <div style={{ borderLeft: '1px solid #e5e7eb', paddingLeft: '1rem', marginRight: '1rem' }}>
              <button 
                onClick={() => setIsFAQModalOpen(true)}
                style={{ 
                  padding: '0.5rem 1rem', 
                  backgroundColor: '#2563eb', 
                  color: 'white', 
                  borderRadius: '0.375rem',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '0.875rem'
                }}
              >
                FAQ
              </button>
            </div>
            
            {/* File selection */}
            <div style={{ borderLeft: '1px solid #e5e7eb', paddingLeft: '1rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input 
                  type="file"
                  accept="application/pdf"
                  onChange={handleFileChange}
                  style={{ 
                    width: '7rem',
                    fontSize: '0.75rem', 
                    color: '#6b7280'
                  }}
                />
              </label>
            </div>
          </div>
        </div>
      )}

      <div style={{ 
        width: '100%', 
        maxWidth: '64rem', 
        backgroundColor: '#f3f4f6', 
        borderRadius: '0.5rem', 
        padding: '1.5rem', 
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
      }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          marginBottom: '1rem'
        }}>
          {!pdfFile && (
            <button 
              onClick={() => setIsFAQModalOpen(true)}
              style={{ 
                padding: '0.5rem 1rem', 
                backgroundColor: '#2563eb', 
                color: 'white', 
                borderRadius: '0.375rem',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.875rem'
              }}
            >
              FAQ
            </button>
          )}
        </div>
        
        {!pdfFile && (
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#4b5563' }}>
              Choose PDF File:
              <input 
                type="file"
                accept="application/pdf"
                onChange={handleFileChange}
                style={{ 
                  marginTop: '0.5rem', 
                  display: 'block', 
                  width: '100%', 
                  fontSize: '0.875rem', 
                  color: '#6b7280'
                }}
              />
            </label>
          </div>
        )}
        
        {error && (
          <div style={{ 
            marginBottom: '1rem', 
            padding: '1rem', 
            color: '#b91c1c', 
            backgroundColor: '#fee2e2', 
            borderRadius: '0.375rem' 
          }}>
            {error}
          </div>
        )}
        
        {pdfFile && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ 
              marginBottom: '1rem', 
              border: '1px solid #d1d5db', 
              borderRadius: '0.5rem', 
              overflow: 'auto', 
              backgroundColor: 'white' 
            }}>
              <Document
                file={pdfFile}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
                style={{ margin: '0 auto' }}
                externalLinkTarget="_blank"
                externalLinkRel="noopener noreferrer"
              >
                <Page 
                  pageNumber={pageNumber} 
                  scale={scale}
                  renderTextLayer={true}
                  renderAnnotationLayer={true}
                  externalLinkTarget="_blank"
                />
              </Document>
            </div>
          </div>
        )}
      </div>
      
      {/* FAQ Modal - Using hardcoded version for Vercel */}
      <HardcodedFAQModal 
        isOpen={isFAQModalOpen} 
        onClose={() => setIsFAQModalOpen(false)} 
      />
    </div>
  );
};

export default AccessiblePDFViewer;