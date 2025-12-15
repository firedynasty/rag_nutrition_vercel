# Preloaded Reports System Documentation

This document explains how to implement a preloading system that converts folders of `.txt` and `.md` files into JSON files for a web app to consume.

## Overview

The system consists of:
1. **Input**: A folder structure containing subfolders with `.txt`/`.md` files
2. **Script**: A Python script that processes the folders
3. **Output**: JSON files that a web app can fetch

## Folder Structure

```
your-project/
├── preloaded_reports/          # Input folder (configurable)
│   ├── topic_one/
│   │   ├── file1.md
│   │   ├── file2.txt
│   │   └── notes.md
│   ├── topic_two/
│   │   ├── document.txt
│   │   └── summary.md
│   └── topic_three/
│       └── content.txt
│
├── public/preloaded/           # Output folder (configurable)
│   ├── index.json              # Lists all available folders
│   ├── topic_one.json          # Contains all files from topic_one/
│   ├── topic_two.json
│   └── topic_three.json
│
└── generate_preloaded.py       # The processing script
```

## Output Format

### index.json
Lists all processed folders with file counts:
```json
{
  "folders": [
    { "name": "topic_one", "fileCount": 3 },
    { "name": "topic_two", "fileCount": 2 },
    { "name": "topic_three", "fileCount": 1 }
  ]
}
```

### {folder_name}.json
Contains all file contents from that folder:
```json
{
  "file1.md": "# Title\n\nContent of file1...",
  "file2.txt": "Plain text content...",
  "notes.md": "More markdown content..."
}
```

## The Python Script

Copy this script to your project as `generate_preloaded.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate preloaded reports JSON files.

Scans input directory for folders containing .txt and .md files,
and generates JSON files in the output directory.

Usage:
    python generate_preloaded.py
    python generate_preloaded.py -i ./my_reports -o ./public/data
"""

import os
import json
import argparse

def process_file(file_path):
    """Read a file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"  Warning: Could not read {file_path}: {e}")
        return None

def generate_preloaded_reports(input_dir='./preloaded_reports', output_dir='./public/preloaded'):
    """Generate JSON files from folders containing text/markdown files."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    folders = []
    folder_entries = os.listdir(input_dir)

    for folder_name in sorted(folder_entries):
        folder_path = os.path.join(input_dir, folder_name)

        # Skip non-directories and hidden folders
        if not os.path.isdir(folder_path) or folder_name.startswith('.'):
            continue

        print(f"Processing folder: {folder_name}")

        # Get all .txt and .md files in the folder
        files_data = {}
        file_list = os.listdir(folder_path)
        text_files = [f for f in file_list if f.endswith('.txt') or f.endswith('.md')]
        text_files.sort(key=lambda x: x.lower())

        file_count = 0
        for filename in text_files:
            file_path = os.path.join(folder_path, filename)
            content = process_file(file_path)

            if content is not None:
                files_data[filename] = content
                file_count += 1
                print(f"  - {filename} ({len(content):,} chars)")

        if file_count > 0:
            # Save folder contents to JSON
            folder_json_path = os.path.join(output_dir, f"{folder_name}.json")
            with open(folder_json_path, 'w', encoding='utf-8') as f:
                json.dump(files_data, f, ensure_ascii=False, indent=2)
            print(f"  Saved to {folder_json_path}")

            folders.append({
                'name': folder_name,
                'fileCount': file_count
            })
        else:
            print(f"  No .txt or .md files found, skipping.")

    # Save index of all folders
    index_path = os.path.join(output_dir, 'index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({'folders': folders}, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated index.json with {len(folders)} folder(s)")
    print(f"Output directory: {output_dir}")

    return folders

def main():
    parser = argparse.ArgumentParser(description='Generate preloaded reports JSON files.')
    parser.add_argument('-i', '--input', default='./preloaded_reports',
                        help='Input directory containing report folders (default: ./preloaded_reports)')
    parser.add_argument('-o', '--output', default='./public/preloaded',
                        help='Output directory for JSON files (default: ./public/preloaded)')

    args = parser.parse_args()
    generate_preloaded_reports(args.input, args.output)

if __name__ == "__main__":
    main()
```

## Usage

### Basic Usage
```bash
python generate_preloaded.py
```
Uses defaults: `./preloaded_reports` → `./public/preloaded`

### Custom Directories
```bash
python generate_preloaded.py -i ./my_content -o ./dist/data
```

## Consuming in JavaScript/React

### Fetching the Index
```javascript
async function loadPreloadedIndex() {
  const response = await fetch('/preloaded/index.json');
  const data = await response.json();
  return data.folders; // [{ name: "topic_one", fileCount: 3 }, ...]
}
```

### Fetching a Folder's Contents
```javascript
async function loadFolder(folderName) {
  const response = await fetch(`/preloaded/${folderName}.json`);
  const files = await response.json();
  // files = { "file1.md": "content...", "file2.txt": "content..." }
  return files;
}
```

### Example React Component
```jsx
import { useState, useEffect } from 'react';

function PreloadedSelector() {
  const [folders, setFolders] = useState([]);
  const [selectedFolder, setSelectedFolder] = useState('');
  const [files, setFiles] = useState({});

  // Load index on mount
  useEffect(() => {
    fetch('/preloaded/index.json')
      .then(res => res.json())
      .then(data => setFolders(data.folders))
      .catch(err => console.log('No preloaded content'));
  }, []);

  // Load folder contents when selected
  useEffect(() => {
    if (selectedFolder) {
      fetch(`/preloaded/${selectedFolder}.json`)
        .then(res => res.json())
        .then(data => setFiles(data));
    }
  }, [selectedFolder]);

  return (
    <div>
      <select
        value={selectedFolder}
        onChange={e => setSelectedFolder(e.target.value)}
      >
        <option value="">Select a folder...</option>
        {folders.map(f => (
          <option key={f.name} value={f.name}>
            {f.name} ({f.fileCount} files)
          </option>
        ))}
      </select>

      {Object.entries(files).map(([filename, content]) => (
        <div key={filename}>
          <h3>{filename}</h3>
          <pre>{content}</pre>
        </div>
      ))}
    </div>
  );
}
```

## Workflow Summary

1. **Create input folder structure**:
   ```bash
   mkdir -p preloaded_reports/my_topic
   ```

2. **Add your `.md` and `.txt` files**:
   ```bash
   # Add files to preloaded_reports/my_topic/
   ```

3. **Run the generator**:
   ```bash
   python generate_preloaded.py
   ```

4. **Start your app** - the JSON files are now available at `/preloaded/`

## Notes

- Hidden folders (starting with `.`) are skipped
- Only `.txt` and `.md` files are processed
- Files are sorted alphabetically (case-insensitive)
- UTF-8 encoding is used for all file operations
- The script creates the output directory if it doesn't exist
- Re-running the script overwrites existing JSON files
