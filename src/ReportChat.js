import React, { useState, useRef, useEffect } from 'react';
import { marked } from 'marked';

// Convert markdown to plain text
const md2plain = (md) => {
  if (!md) return '';
  return md
    .replace(/^#{1,6}\s+/gm, '')           // Headers
    .replace(/\*\*([^*]+)\*\*/g, '$1')     // Bold
    .replace(/\*([^*]+)\*/g, '$1')         // Italic
    .replace(/__([^_]+)__/g, '$1')         // Bold alt
    .replace(/_([^_]+)_/g, '$1')           // Italic alt
    .replace(/~~([^~]+)~~/g, '$1')         // Strikethrough
    .replace(/`{3}[\s\S]*?`{3}/g, '')      // Code blocks
    .replace(/`([^`]+)`/g, '$1')           // Inline code
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Links
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1') // Images
    .replace(/^\s*[-*+]\s+/gm, '• ')       // Unordered lists
    .replace(/^\s*\d+\.\s+/gm, '')         // Ordered lists
    .replace(/^>\s+/gm, '')                // Blockquotes
    .replace(/^---+$/gm, '')               // Horizontal rules
    .replace(/\n{3,}/g, '\n\n')            // Multiple newlines
    .trim();
};

const ReportChat = () => {
  // API and model state
  const [aiProvider, setAiProvider] = useState('ChatGPT');
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('gpt-4o-mini');
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [ragEnabled, setRagEnabled] = useState(true); // Default: RAG enabled
  const [ragContext, setRagContext] = useState(''); // Store last RAG context
  const [showRagContext, setShowRagContext] = useState(false);
  const [useSharedKey, setUseSharedKey] = useState(false);
  const [accessCode, setAccessCode] = useState('');
  const [selectedRole, setSelectedRole] = useState('nutrition_rag'); // Default: Nutrition RAG role

  // File state
  const [preloadedFiles, setPreloadedFiles] = useState([]); // list of filenames
  const [selectedFile, setSelectedFile] = useState('');
  const [allFileContents, setAllFileContents] = useState({}); // {filename: content}
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [sendAllFiles, setSendAllFiles] = useState(false); // false = current file only, true = all files

  // Modal state
  const [showPromptModal, setShowPromptModal] = useState(false);

  // Chat state
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Refs
  const chatContainerRef = useRef(null);
  const folderInputRef = useRef(null);

  // Available roles
  const roles = {
    'default': { name: 'Default (No Role)', prompt: null },
    'socratic_coach': {
      name: 'Socratic Coach',
      prompt: `You are a Socratic coach helping the user critically examine and challenge the information in the document.

YOUR ROLE: Help the user question assumptions, find weaknesses, and think critically about what they're reading. Don't accept claims at face value - probe them.

WHEN THE USER ASKS ABOUT THE CONTENT:
1. Point out what claims are being made
2. Ask: "What evidence supports this?" or "What's the counter-argument?"
3. Challenge weak reasoning or unsupported assertions
4. Help them distinguish fact from opinion

EXAMPLE RESPONSES:
- "This document claims X. What evidence do you see for that? What might be missing?"
- "That's an interesting assertion. Can you think of situations where it wouldn't hold?"
- "Before accepting this, what questions should we ask?"

TONE: Curious, challenging, thought-provoking. Push them to think deeper.

KEEP RESPONSES SHORT (under 150 words). End with a probing question.`
    },
    'nutrition_rag': {
      name: 'Nutrition RAG Assistant',
      prompt: `You are a helpful nutrition and health assistant with access to a knowledge base of nutrition research articles.

When answering questions:
1. Base your answers on the provided context from retrieved articles
2. Cite the source titles/URLs when referencing specific information
3. If the context doesn't contain relevant information, say so and offer general guidance
4. Be accurate and don't make claims not supported by the provided sources
5. If multiple sources have different perspectives, acknowledge this

Use the RAG context to provide informed, evidence-based answers about nutrition and health topics.`
    },
  };

  // Available models
  const anthropicModels = {
    'Claude 3.5 Haiku': 'claude-3-5-haiku-20241022',
    'Claude 3.5 Sonnet': 'claude-3-5-sonnet-20241022',
    'Claude Sonnet 4': 'claude-sonnet-4-20250514',
    'Claude Opus 4.5': 'claude-opus-4-5-20251101',
  };

  const openaiModels = {
    'GPT-4o': 'gpt-4o',
    'GPT-4o Mini': 'gpt-4o-mini',
    'GPT-4 Turbo': 'gpt-4-turbo',
    'GPT-3.5 Turbo': 'gpt-3.5-turbo',
    'o1': 'o1',
    'o1 Mini': 'o1-mini',
  };

  const models = aiProvider === 'ChatGPT' ? openaiModels : anthropicModels;

  // Reset model when provider changes
  useEffect(() => {
    if (aiProvider === 'ChatGPT') {
      setSelectedModel('gpt-4o-mini');
    } else {
      setSelectedModel('claude-3-5-haiku-20241022');
    }
  }, [aiProvider]);

  // Fetch preloaded files list and contents on mount
  useEffect(() => {
    const fetchPreloadedData = async () => {
      setIsLoadingFiles(true);
      try {
        // Fetch index.json for file list
        const indexResponse = await fetch('/preloaded/index.json');
        if (indexResponse.ok) {
          const indexData = await indexResponse.json();
          setPreloadedFiles(indexData.files || []);
        }

        // Fetch reports.json for all file contents
        const reportsResponse = await fetch('/preloaded/reports.json');
        if (reportsResponse.ok) {
          const reportsData = await reportsResponse.json();
          setAllFileContents(reportsData);
        }
      } catch (err) {
        console.log('Error loading preloaded data:', err);
      } finally {
        setIsLoadingFiles(false);
      }
    };
    fetchPreloadedData();
  }, []);

  // Auto-scroll chat to bottom when new messages arrive
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Check if file is markdown
  const isMarkdown = (filename) => {
    return filename?.toLowerCase().endsWith('.md');
  };

  // Render markdown content
  const renderMarkdown = (content) => {
    if (!content) return '';
    return marked.parse(content);
  };

  // Get selected file content
  const getSelectedFileContent = () => {
    if (!selectedFile || !allFileContents[selectedFile]) return null;
    return allFileContents[selectedFile];
  };

  // Navigate to next/previous report
  const navigateReport = (direction) => {
    if (preloadedFiles.length === 0) return;

    const currentIndex = preloadedFiles.indexOf(selectedFile);
    let newIndex;

    if (direction === 'next') {
      newIndex = currentIndex < preloadedFiles.length - 1 ? currentIndex + 1 : 0;
    } else {
      newIndex = currentIndex > 0 ? currentIndex - 1 : preloadedFiles.length - 1;
    }

    setSelectedFile(preloadedFiles[newIndex]);
  };

  // Build system prompt with role and selected file context
  const buildSystemPrompt = () => {
    const parts = [];

    // Add role prompt if not default
    const rolePrompt = roles[selectedRole]?.prompt;
    if (rolePrompt) {
      parts.push(rolePrompt);
    }

    // Add file content(s) as context
    if (sendAllFiles && Object.keys(allFileContents).length > 0) {
      // Send ALL files concatenated
      const allFilesContext = Object.entries(allFileContents)
        .map(([filename, content]) => `--- FILE: ${filename} ---\n${content}`)
        .join('\n\n---\n\n');

      const fileContext = `You have access to the following ${Object.keys(allFileContents).length} documents. Use them to answer questions and provide analysis.

${allFilesContext}
`;
      parts.push(fileContext);
    } else {
      // Send only the selected file (default)
      const fileContent = getSelectedFileContent();
      if (fileContent) {
        const fileContext = `You have access to the following document. Use it to answer questions and provide analysis.

--- FILE: ${selectedFile} ---
${fileContent}
`;
        parts.push(fileContext);
      }
    }

    return parts.length > 0 ? parts.join('\n\n') : null;
  };

  // Send message to API
  const sendMessage = async () => {
    const needsApiKey = !(aiProvider === 'ChatGPT' && useSharedKey);
    if (!inputValue.trim()) return;
    if (needsApiKey && !apiKey) {
      alert(`Please enter your ${aiProvider === 'ChatGPT' ? 'OpenAI' : 'Anthropic'} API key`);
      return;
    }

    const userMessage = { role: 'user', content: inputValue };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputValue('');
    setIsLoading(true);

    try {
      let systemPrompt = buildSystemPrompt();
      let assistantMessage;

      // If RAG is enabled, first retrieve context from the knowledge base
      if (ragEnabled) {
        try {
          const ragResponse = await fetch('/api/rag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query: inputValue,
              retrieveOnly: true, // Just get context, we'll call LLM ourselves
              apiKey: useSharedKey ? null : apiKey,
              accessCode: useSharedKey ? accessCode : null,
            }),
          });

          if (ragResponse.ok) {
            const ragData = await ragResponse.json();
            setRagContext(ragData.context);

            // Prepend RAG context to system prompt
            const ragPrefix = `[RAG Context from Nutrition Knowledge Base]
${ragData.context}
[End RAG Context]

`;
            systemPrompt = ragPrefix + (systemPrompt || roles['nutrition_rag'].prompt);
          } else {
            const errorData = await ragResponse.json();
            console.error('RAG retrieval failed:', errorData.error);
            // Continue without RAG context
          }
        } catch (ragError) {
          console.error('RAG API error:', ragError);
          // Continue without RAG context
        }
      }

      if (aiProvider === 'ChatGPT') {
        const openaiMessages = [];
        if (systemPrompt) {
          openaiMessages.push({ role: 'system', content: systemPrompt });
        }
        openaiMessages.push(...newMessages.map(m => ({ role: m.role, content: m.content })));

        const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

        if (isLocalhost && !useSharedKey) {
          let response;

          if (webSearchEnabled) {
            response = await fetch('https://api.openai.com/v1/responses', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`,
              },
              body: JSON.stringify({
                model: selectedModel,
                tools: [{ type: 'web_search' }],
                tool_choice: 'auto',
                input: newMessages[newMessages.length - 1].content,
              }),
            });

            if (!response.ok) {
              const errorData = await response.json();
              throw new Error(errorData.error?.message || 'API request failed');
            }

            const data = await response.json();
            let responseText = data.output_text;
            if (!responseText && data.output) {
              for (const item of data.output) {
                if (item.type === 'message' && item.content) {
                  for (const content of item.content) {
                    if (content.type === 'output_text' && content.text) {
                      responseText = content.text;
                      break;
                    }
                  }
                }
                if (responseText) break;
              }
            }

            assistantMessage = {
              role: 'assistant',
              content: responseText || 'No response from web search',
            };
          } else {
            response = await fetch('https://api.openai.com/v1/chat/completions', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`,
              },
              body: JSON.stringify({
                model: selectedModel,
                max_tokens: 4096,
                messages: openaiMessages,
              }),
            });

            if (!response.ok) {
              const errorData = await response.json();
              throw new Error(errorData.error?.message || 'API request failed');
            }

            const data = await response.json();
            assistantMessage = {
              role: 'assistant',
              content: data.choices[0].message.content,
            };
          }
        } else {
          const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              messages: openaiMessages,
              model: selectedModel,
              userApiKey: useSharedKey ? null : apiKey,
              accessCode: useSharedKey ? accessCode : null,
              webSearch: webSearchEnabled,
            }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'API request failed');
          }

          const data = await response.json();
          assistantMessage = {
            role: 'assistant',
            content: data.content,
          };
        }
      } else {
        const requestBody = {
          model: selectedModel,
          max_tokens: 4096,
          messages: newMessages.map(m => ({ role: m.role, content: m.content })),
        };

        if (systemPrompt) {
          requestBody.system = systemPrompt;
        }

        const response = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey,
            'anthropic-version': '2023-06-01',
            'anthropic-dangerous-direct-browser-access': 'true',
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error?.message || 'API request failed');
        }

        const data = await response.json();
        assistantMessage = {
          role: 'assistant',
          content: data.content[0].text,
        };
      }

      setMessages([...newMessages, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      setMessages([...newMessages, {
        role: 'assistant',
        content: `Error: ${error.message}`,
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Clear chat
  const clearChat = () => {
    if (messages.length === 0 || window.confirm('Clear all messages?')) {
      setMessages([]);
      setRagContext('');
    }
  };

  // Copy chat to clipboard
  const copyChat = async () => {
    if (messages.length === 0) {
      alert('No messages to copy');
      return;
    }
    const chatText = messages
      .map(m => `${m.role === 'user' ? 'You' : 'Assistant'}:\n${m.content}`)
      .join('\n\n---\n\n');
    try {
      await navigator.clipboard.writeText(chatText);
      alert('Chat copied to clipboard!');
    } catch (err) {
      console.error('Failed to copy:', err);
      alert('Failed to copy chat');
    }
  };

  // Handle folder selection - load .txt and .md files
  const handleFolderSelect = async (e) => {
    const files = Array.from(e.target.files);

    // Filter for .txt and .md files only
    const textFiles = files.filter(file => {
      const name = file.name.toLowerCase();
      return name.endsWith('.txt') || name.endsWith('.md');
    });

    if (textFiles.length === 0) {
      alert('No .txt or .md files found in the selected folder');
      return;
    }

    setIsLoadingFiles(true);

    const newFileContents = {};
    const newFileNames = [];

    // Read each file
    for (const file of textFiles) {
      try {
        const content = await file.text();
        newFileContents[file.name] = content;
        newFileNames.push(file.name);
      } catch (err) {
        console.error(`Error reading ${file.name}:`, err);
      }
    }

    // Sort filenames alphabetically
    newFileNames.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));

    // Update state
    setPreloadedFiles(newFileNames);
    setAllFileContents(newFileContents);
    setSelectedFile(''); // Reset selection
    setIsLoadingFiles(false);

    // Reset the input so the same folder can be selected again
    e.target.value = '';
  };

  const fileContent = getSelectedFileContent();

  return (
    <div style={styles.container}>
      {/* Sidebar */}
      <div style={styles.sidebar}>
        <h2 style={styles.sidebarTitle}>Report Chat</h2>

        {/* AI Provider Toggle */}
        <div style={styles.section}>
          <label style={styles.label}>AI Provider:</label>
          <div style={styles.providerToggle}>
            <button
              onClick={() => setAiProvider('ChatGPT')}
              style={{
                ...styles.providerBtn,
                ...(aiProvider === 'ChatGPT' ? styles.providerBtnActive : {}),
              }}
            >
              ChatGPT
            </button>
            <button
              onClick={() => setAiProvider('Anthropic')}
              style={{
                ...styles.providerBtn,
                ...(aiProvider === 'Anthropic' ? styles.providerBtnActive : {}),
              }}
            >
              Claude
            </button>
          </div>
          {aiProvider === 'ChatGPT' && (
            <button
              onClick={() => {
                const code = prompt("Enter access code for Stanley's key:");
                if (code) {
                  setAccessCode(code);
                  setUseSharedKey(true);
                  setApiKey('');
                }
              }}
              style={{
                ...styles.button,
                marginTop: '8px',
                background: useSharedKey ? '#28a745' : '#6c757d',
              }}
            >
              {useSharedKey ? "✓ Using Stanley's Key" : "Use Stanley's Key"}
            </button>
          )}
        </div>

        {/* API Key Input */}
        <div style={styles.section}>
          <label style={styles.label}>{aiProvider === 'ChatGPT' ? 'OpenAI' : 'Anthropic'} API Key:</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={aiProvider === 'ChatGPT' ? 'sk-...' : 'sk-ant-...'}
            style={styles.input}
          />
        </div>

        {/* Model Selection */}
        <div style={styles.section}>
          <label style={styles.label}>{aiProvider === 'ChatGPT' ? 'OpenAI' : 'Claude'} Model:</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={styles.select}
          >
            {Object.entries(models).map(([name, id]) => (
              <option key={id} value={id}>{name}</option>
            ))}
          </select>
        </div>

        {/* Web Search Toggle - Only for ChatGPT */}
        {aiProvider === 'ChatGPT' && (
          <div style={styles.section}>
            <label style={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={webSearchEnabled}
                onChange={(e) => setWebSearchEnabled(e.target.checked)}
                style={styles.checkbox}
              />
              Enable Web Search
            </label>
          </div>
        )}

        {/* RAG Toggle - Nutrition Knowledge Base */}
        <div style={styles.section}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={ragEnabled}
              onChange={(e) => {
                setRagEnabled(e.target.checked);
                if (e.target.checked) {
                  setSelectedRole('nutrition_rag');
                }
              }}
              style={styles.checkbox}
            />
            <span style={{ color: ragEnabled ? '#28a745' : '#fff' }}>
              🥦 Nutrition RAG
            </span>
          </label>
          {ragEnabled && ragContext && (
            <button
              onClick={() => setShowRagContext(!showRagContext)}
              style={{ ...styles.button, marginTop: '8px', background: '#6c757d', fontSize: '12px' }}
            >
              {showRagContext ? 'Hide Context' : 'Show RAG Context'}
            </button>
          )}
        </div>

        {/* Role Selection */}
        <div style={styles.section}>
          <label style={styles.label}>Role:</label>
          <select
            value={selectedRole}
            onChange={(e) => setSelectedRole(e.target.value)}
            style={styles.select}
          >
            {Object.entries(roles).map(([key, role]) => (
              <option key={key} value={key}>{role.name}</option>
            ))}
          </select>
          {roles[selectedRole]?.prompt && (
            <button
              onClick={() => setShowPromptModal(true)}
              style={{ ...styles.button, marginTop: '8px', background: '#6c757d' }}
            >
              Show Prompt
            </button>
          )}
        </div>

        {/* File Selection Dropdown */}
        <div style={styles.section}>
          <label style={styles.label}>Select Report:</label>
          <select
            value={selectedFile}
            onChange={(e) => setSelectedFile(e.target.value)}
            style={styles.select}
            disabled={isLoadingFiles}
          >
            <option value="">-- Select a file --</option>
            {preloadedFiles.map((filename) => (
              <option key={filename} value={filename}>
                {filename}
              </option>
            ))}
          </select>

          {/* Navigation buttons */}
          <div style={styles.navButtons}>
            <button
              onClick={() => navigateReport('prev')}
              style={styles.navBtn}
              disabled={preloadedFiles.length === 0}
              title="Previous report"
            >
              ◀ Prev
            </button>
            <span style={styles.navCounter}>
              {selectedFile ? `${preloadedFiles.indexOf(selectedFile) + 1}/${preloadedFiles.length}` : `0/${preloadedFiles.length}`}
            </span>
            <button
              onClick={() => navigateReport('next')}
              style={styles.navBtn}
              disabled={preloadedFiles.length === 0}
              title="Next report"
            >
              Next ▶
            </button>
          </div>

          {/* Hidden folder input */}
          <input
            type="file"
            ref={folderInputRef}
            onChange={handleFolderSelect}
            style={{ display: 'none' }}
            webkitdirectory=""
            directory=""
            multiple
          />

          {/* Load Folder Button */}
          <button
            onClick={() => folderInputRef.current?.click()}
            style={{ ...styles.button, marginTop: '8px', background: '#28a745' }}
            disabled={isLoadingFiles}
          >
            {isLoadingFiles ? 'Loading...' : '📂 Load Folder'}
          </button>

          {isLoadingFiles && (
            <p style={styles.charCount}>Loading...</p>
          )}
        </div>

        {/* Credit */}
        <div style={styles.credit}>
          Credit to <a href="https://github.com/alexkolo/rag_nutrition_facts_blog" target="_blank" rel="noopener noreferrer" style={styles.creditLink}>alexkolo/rag_nutrition_facts_blog</a>
        </div>

        {/* Send Mode Toggle */}
        <div style={styles.section}>
          <label style={styles.label}>Send to Chat:</label>
          <div style={styles.toggleContainer}>
            <span style={{ fontSize: '12px', color: !sendAllFiles ? '#4da6ff' : '#888', fontWeight: !sendAllFiles ? 'bold' : 'normal' }}>
              📄 Current
            </span>
            <div
              onClick={() => setSendAllFiles(!sendAllFiles)}
              style={{
                ...styles.sliderTrack,
                backgroundColor: sendAllFiles ? '#4da6ff' : '#6c757d',
              }}
            >
              <div
                style={{
                  ...styles.sliderKnob,
                  transform: sendAllFiles ? 'translateX(22px)' : 'translateX(0)',
                }}
              />
            </div>
            <span style={{ fontSize: '12px', color: sendAllFiles ? '#4da6ff' : '#888', fontWeight: sendAllFiles ? 'bold' : 'normal' }}>
              📚 All Files
            </span>
          </div>
          <p style={styles.charCount}>
            {sendAllFiles
              ? `Sending all ${Object.keys(allFileContents).length} files`
              : selectedFile ? `Sending: ${selectedFile}` : 'No file selected'}
          </p>
        </div>
      </div>

      {/* Main Area - Side by Side */}
      <div style={styles.mainArea}>
        {/* File Content Panel (Left) */}
        <div style={styles.filePanel}>
          <div style={styles.filePanelHeader}>
            {selectedFile || 'No file selected'}
          </div>
          <div style={styles.filePanelContent}>
            {fileContent ? (
              isMarkdown(selectedFile) ? (
                <div
                  style={styles.markdownContent}
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(fileContent) }}
                />
              ) : (
                <pre style={styles.preContent}>{fileContent}</pre>
              )
            ) : (
              <div style={styles.emptyState}>
                <p>Select a file from the dropdown to view its contents</p>
              </div>
            )}
          </div>
        </div>

        {/* Chat Panel (Right) */}
        <div style={styles.chatPanel}>
          <div style={styles.chatPanelHeader}>
            <span>Chat with {aiProvider === 'ChatGPT' ? 'ChatGPT' : 'Claude'}</span>
            <div style={styles.chatHeaderButtons}>
              <button onClick={copyChat} style={styles.chatHeaderBtn} title="Copy chat">
                📋 Copy
              </button>
              <button onClick={clearChat} style={styles.chatHeaderBtn} title="Clear chat">
                🗑️ Clear
              </button>
            </div>
          </div>

          {/* Chat Messages */}
          <div ref={chatContainerRef} style={styles.chatContainer}>
            {messages.length === 0 ? (
              <div style={styles.emptyState}>
                <p>Select a file and start chatting!</p>
              </div>
            ) : (
              messages.map((msg, index) => (
                <div
                  key={index}
                  style={{
                    ...styles.message,
                    ...(msg.role === 'user' ? styles.userMessage : styles.assistantMessage),
                  }}
                >
                  <div style={styles.messageRole}>
                    {msg.role === 'user' ? 'You' : (aiProvider === 'ChatGPT' ? 'ChatGPT' : 'Claude')}
                  </div>
                  <div style={styles.messageContent}>
                    {msg.role === 'user' ? msg.content : md2plain(msg.content)}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div style={{ ...styles.message, ...styles.assistantMessage }}>
                <div style={styles.messageRole}>{aiProvider === 'ChatGPT' ? 'ChatGPT' : 'Claude'}</div>
                <div style={styles.messageContent}>Thinking...</div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div style={styles.inputArea}>
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message... (Enter to send)"
              style={styles.textarea}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !inputValue.trim()}
              style={{
                ...styles.sendButton,
                opacity: isLoading || !inputValue.trim() ? 0.6 : 1,
              }}
            >
              Send
            </button>
          </div>
        </div>
      </div>

      {/* Prompt Modal */}
      {showPromptModal && (
        <div style={styles.modalOverlay} onClick={() => setShowPromptModal(false)}>
          <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <h3 style={styles.modalTitle}>{roles[selectedRole]?.name} - System Prompt</h3>
              <button
                onClick={() => setShowPromptModal(false)}
                style={styles.modalCloseBtn}
              >
                OK
              </button>
            </div>
            <div style={styles.modalContent}>
              <pre style={styles.promptText}>{roles[selectedRole]?.prompt}</pre>
            </div>
          </div>
        </div>
      )}

      {/* RAG Context Modal */}
      {showRagContext && ragContext && (
        <div style={styles.modalOverlay} onClick={() => setShowRagContext(false)}>
          <div style={{ ...styles.modal, width: '800px' }} onClick={(e) => e.stopPropagation()}>
            <div style={{ ...styles.modalHeader, background: '#28a745' }}>
              <h3 style={styles.modalTitle}>🥦 RAG Context - Retrieved Articles</h3>
              <button
                onClick={() => setShowRagContext(false)}
                style={styles.modalCloseBtn}
              >
                Close
              </button>
            </div>
            <div style={styles.modalContent}>
              <pre style={styles.promptText}>{ragContext}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Styles
const styles = {
  container: {
    display: 'flex',
    height: '100vh',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    background: '#f5f5f5',
  },
  sidebar: {
    width: '280px',
    background: '#1a1a2e',
    color: '#fff',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    overflowY: 'auto',
    flexShrink: 0,
  },
  sidebarTitle: {
    margin: '0 0 10px 0',
    fontSize: '24px',
    fontWeight: '600',
    color: '#4da6ff',
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  providerToggle: {
    display: 'flex',
    gap: '0',
    borderRadius: '6px',
    overflow: 'hidden',
  },
  providerBtn: {
    flex: 1,
    padding: '10px 12px',
    border: 'none',
    fontSize: '14px',
    fontWeight: '500',
    background: '#2d2d44',
    color: '#888',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  providerBtnActive: {
    background: '#4da6ff',
    color: '#fff',
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '14px',
    color: '#fff',
    cursor: 'pointer',
  },
  toggleContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '8px',
    background: '#2d2d44',
    borderRadius: '6px',
  },
  sliderTrack: {
    position: 'relative',
    width: '44px',
    height: '22px',
    borderRadius: '22px',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
  },
  sliderKnob: {
    position: 'absolute',
    top: '2px',
    left: '2px',
    width: '18px',
    height: '18px',
    borderRadius: '50%',
    backgroundColor: '#fff',
    boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
    transition: 'transform 0.3s',
  },
  navButtons: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '8px',
  },
  navBtn: {
    flex: 1,
    padding: '8px 12px',
    borderRadius: '6px',
    border: 'none',
    fontSize: '12px',
    fontWeight: '500',
    background: '#2d2d44',
    color: '#fff',
    cursor: 'pointer',
    transition: 'background 0.2s',
  },
  navCounter: {
    fontSize: '12px',
    color: '#888',
    minWidth: '50px',
    textAlign: 'center',
  },
  checkbox: {
    width: '18px',
    height: '18px',
    cursor: 'pointer',
  },
  label: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#aaa',
  },
  input: {
    padding: '10px 12px',
    borderRadius: '6px',
    border: 'none',
    fontSize: '14px',
    background: '#2d2d44',
    color: '#fff',
  },
  select: {
    padding: '10px 12px',
    borderRadius: '6px',
    border: 'none',
    fontSize: '14px',
    background: '#2d2d44',
    color: '#fff',
    cursor: 'pointer',
  },
  button: {
    padding: '10px 16px',
    borderRadius: '6px',
    border: 'none',
    fontSize: '14px',
    fontWeight: '500',
    background: '#4da6ff',
    color: '#fff',
    cursor: 'pointer',
    transition: 'background 0.2s',
  },
  charCount: {
    fontSize: '12px',
    color: '#888',
    margin: '4px 0 0 0',
  },
  credit: {
    fontSize: '11px',
    color: '#666',
    textAlign: 'center',
    padding: '8px',
    borderTop: '1px solid #333',
  },
  creditLink: {
    color: '#4da6ff',
    textDecoration: 'none',
  },
  mainArea: {
    flex: 1,
    display: 'flex',
    overflow: 'hidden',
  },
  filePanel: {
    width: '50%',
    minWidth: '50%',
    maxWidth: '50%',
    display: 'flex',
    flexDirection: 'column',
    borderRight: '1px solid #ddd',
    background: '#fff',
  },
  filePanelHeader: {
    padding: '15px 20px',
    background: '#1a1a2e',
    color: '#4da6ff',
    fontSize: '14px',
    fontWeight: '600',
    borderBottom: '1px solid #333',
  },
  filePanelContent: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px',
  },
  chatPanel: {
    width: '50%',
    minWidth: '50%',
    maxWidth: '50%',
    display: 'flex',
    flexDirection: 'column',
    background: '#fff',
  },
  chatPanelHeader: {
    padding: '15px 20px',
    background: '#1a1a2e',
    color: '#4da6ff',
    fontSize: '14px',
    fontWeight: '600',
    borderBottom: '1px solid #333',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  chatHeaderButtons: {
    display: 'flex',
    gap: '8px',
  },
  chatHeaderBtn: {
    padding: '6px 12px',
    borderRadius: '4px',
    border: 'none',
    fontSize: '12px',
    background: '#2d2d44',
    color: '#fff',
    cursor: 'pointer',
    transition: 'background 0.2s',
  },
  chatContainer: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  emptyState: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#666',
  },
  message: {
    padding: '12px 16px',
    borderRadius: '12px',
    maxWidth: '85%',
  },
  userMessage: {
    alignSelf: 'flex-end',
    background: '#4da6ff',
    color: '#fff',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    background: '#f0f0f0',
    color: '#333',
  },
  messageRole: {
    fontSize: '12px',
    fontWeight: '600',
    marginBottom: '4px',
    opacity: 0.8,
  },
  messageContent: {
    fontSize: '14px',
    lineHeight: '1.5',
    whiteSpace: 'pre-wrap',
  },
  inputArea: {
    padding: '16px 20px',
    borderTop: '1px solid #e0e0e0',
    display: 'flex',
    gap: '12px',
    background: '#fafafa',
  },
  textarea: {
    flex: 1,
    padding: '12px',
    borderRadius: '8px',
    border: '1px solid #ddd',
    fontSize: '14px',
    resize: 'none',
    minHeight: '50px',
    maxHeight: '150px',
    fontFamily: 'inherit',
  },
  sendButton: {
    padding: '12px 24px',
    borderRadius: '8px',
    border: 'none',
    background: '#4da6ff',
    color: '#fff',
    fontSize: '14px',
    fontWeight: '500',
    cursor: 'pointer',
    alignSelf: 'flex-end',
  },
  markdownContent: {
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif",
    lineHeight: '1.6',
    color: '#333',
  },
  preContent: {
    whiteSpace: 'pre-wrap',
    wordWrap: 'break-word',
    margin: 0,
    fontFamily: "'Courier New', monospace",
    fontSize: '14px',
    lineHeight: '1.6',
    color: '#333',
  },
  modalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  modal: {
    background: '#fff',
    borderRadius: '12px',
    width: '600px',
    maxWidth: '90%',
    maxHeight: '80vh',
    display: 'flex',
    flexDirection: 'column',
  },
  modalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 20px',
    borderBottom: '1px solid #eee',
    background: '#1a1a2e',
    borderRadius: '12px 12px 0 0',
  },
  modalTitle: {
    margin: 0,
    fontSize: '18px',
    fontWeight: '600',
    color: '#4da6ff',
  },
  modalCloseBtn: {
    padding: '8px 20px',
    borderRadius: '6px',
    border: 'none',
    background: '#4da6ff',
    color: '#fff',
    fontSize: '14px',
    fontWeight: '500',
    cursor: 'pointer',
  },
  modalContent: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px',
  },
  promptText: {
    whiteSpace: 'pre-wrap',
    wordWrap: 'break-word',
    margin: 0,
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    fontSize: '14px',
    lineHeight: '1.6',
    color: '#333',
  },
};

export default ReportChat;
