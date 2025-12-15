#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import csv
import re
import argparse
import subprocess
import sys
import html

def extract_title_from_file(file_path):
    """Extract a title from the file content or file name."""
    file_name = os.path.basename(file_path)
    
    # Extract page number for elementary_chinese files
    if file_name.startswith("elementary_chinese_pg"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Try to find a title in the content
        lines = content.split('\n')
        for line in lines:
            if '|' in line:
                # Extract the part after the pipe which typically contains the title
                title = line.split('|', 1)[1].strip()
                return f"Elementary Chinese ({file_name}) - {title}"
        
        # If no title found, use the file name
        page_num = file_name.replace("elementary_chinese_pg", "").replace(".txt", "")
        return f"Elementary Chinese (Page {page_num})"
    
    # For other files, just use the filename without extension
    base_name = os.path.splitext(file_name)[0]
    return f"{base_name} ({os.path.splitext(file_name)[1][1:]})"

def detect_local_galleries(content, base_dir):
    """Detect local gallery references and add metadata."""
    gallery_info = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for local gallery patterns
        if any(pattern in line for pattern in ['_img/', '_videos/', '_gallery/', '/images/', '/videos/']):
            # Check if corresponding folder exists
            folder_name = None
            if '_img/' in line:
                folder_name = line.split('_img/')[0].split('/')[-1] + '_img'
            elif '_videos/' in line:
                folder_name = line.split('_videos/')[0].split('/')[-1] + '_videos'
            elif '/images/' in line:
                folder_name = line.split('/images/')[0].split('/')[-1]
            elif '/videos/' in line:
                folder_name = line.split('/videos/')[0].split('/')[-1]
            
            if folder_name:
                folder_path = os.path.join(base_dir, '..', folder_name)
                if os.path.exists(folder_path):
                    gallery_info.append({
                        'type': 'local_gallery',
                        'folder': folder_name,
                        'url_line': line
                    })
    
    return gallery_info

def parse_markdown_images(content):
    """Parse markdown-style images and videos, auto-detecting by file extension."""
    # Pattern to match markdown images/videos: ![alt text](file_path)
    media_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    # Video file extensions
    video_extensions = {'.mp4', '.webm', '.ogg', '.mov', '.avi', '.mkv', '.m4v'}
    
    def replace_media(match):
        alt_text = match.group(1)
        file_path = match.group(2)
        
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in video_extensions:
            # Handle as video
            if file_path.startswith('./videos/'):
                file_path = file_path.replace('./videos/', './folder_w_text/videos/')
            
            filename = file_path.split('/')[-1]  # Get just the filename
            return f'<div class="video-container"><video controls width="400" style="border-radius: 8px; margin: 10px 0;"><source src="{file_path}" type="video/mp4">Your browser does not support the video tag.</video><div style="font-size: 12px; color: #666; margin-top: 5px; text-align: center; font-family: monospace;">{filename}</div></div>'
        else:
            # Handle as image
            if file_path.startswith('./images/'):
                file_path = file_path.replace('./images/', './folder_w_text/images/')
            
            return f'<div class="image-container"><img src="{file_path}" alt="{alt_text}" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;"></div>'
    
    # Replace all markdown media
    processed_content = re.sub(media_pattern, replace_media, content)
    
    return processed_content

def parse_markdown_links(content):
    """Parse markdown-style links and convert them to HTML anchor tags."""
    # Pattern to match markdown links: [text](url)
    link_pattern = r'\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_link(match):
        link_text = match.group(1)
        link_url = match.group(2)
        
        # Create HTML anchor tag with styling
        return f'<a href="{link_url}" target="_blank" style="color: #0066cc; text-decoration: none; border-bottom: 1px dotted #0066cc;">{link_text}</a>'
    
    # Replace all markdown links
    processed_content = re.sub(link_pattern, replace_link, content)
    
    return processed_content

def parse_markdown_headings(content):
    """Parse markdown-style headings and convert them to HTML heading tags."""
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        # Check if line starts with # (heading)
        if line.strip().startswith('#'):
            # Count the number of # characters
            hash_count = 0
            for char in line:
                if char == '#':
                    hash_count += 1
                else:
                    break
            
            # Limit to h1-h6
            if 1 <= hash_count <= 6:
                # Get the heading text (everything after the # characters and any spaces)
                heading_text = line[hash_count:].strip()
                if heading_text:  # Only process if there's actual text
                    processed_lines.append(f'<h{hash_count}>{heading_text}</h{hash_count}>')
                else:
                    processed_lines.append(line)  # Keep original if no text after #
            else:
                processed_lines.append(line)  # Keep original if too many #
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def parse_code_blocks(content):
    """Parse markdown-style code blocks and convert them to HTML with syntax highlighting classes."""
    # Pattern to match code blocks with language specifier
    code_block_pattern = r'```(\w+)?\n(.*?)```'
    
    def replace_code_block(match):
        language = match.group(1) if match.group(1) else 'text'
        code_content = match.group(2)
        
        # Escape HTML entities in the code content
        escaped_code = html.escape(code_content)
        
        # Return HTML with Prism.js classes
        return f'<pre><code class="language-{language}">{escaped_code}</code></pre>'
    
    # Replace all code blocks
    processed_content = re.sub(code_block_pattern, replace_code_block, content, flags=re.DOTALL)
    
    return processed_content

def process_txt_file(file_path, base_dir=None):
    """Process a text file and extract its content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Detect local galleries if base_dir provided
    gallery_info = []
    if base_dir:
        gallery_info = detect_local_galleries(content, base_dir)
    
    # Parse markdown images first
    content = parse_markdown_images(content)
    
    # Parse markdown links
    content = parse_markdown_links(content)
    
    # Parse markdown headings
    content = parse_markdown_headings(content)
    
    # Parse code blocks for syntax highlighting
    content = parse_code_blocks(content)
    
    # Basic processing of content to make it more structured
    lines = content.split('\n')
    processed_lines = []
    
    # Simple formatting for elementary_chinese files to extract vocab
    in_vocab_section = False
    vocab_entries = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # Skip lines with interactive elements (but not if they're in code blocks)
        if "interactive" in stripped_line and not ('<pre><code' in line):
            continue
            
        # Look for vocabulary sections
        if "Chinese" in stripped_line and "Pinyin" in stripped_line or "English" in stripped_line:
            in_vocab_section = True
            continue
            
        if in_vocab_section:
            # Try to extract vocabulary entries
            if re.match(r'^\s*\d+\s*$', stripped_line):  # Skip line numbers
                continue
                
            vocab_entries.append(line)  # Keep original line with spacing
        else:
            # For non-vocab sections, preserve all lines including empty ones
            processed_lines.append(line)
    
    # If we found vocab entries, format them
    if vocab_entries:
        processed_content = "\n".join(vocab_entries)
    else:
        # Join processed lines preserving original line breaks
        processed_content = "\n".join(processed_lines)
    
    # For now, just return the content as string to avoid JS errors
    # Gallery info can be added later if needed
    return processed_content

def process_csv_file(file_path):
    """Process a CSV file and extract its content."""
    content_lines = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Skip empty rows
                    content_lines.append(",".join(row))
    except Exception as e:
        return f"Error reading CSV: {str(e)}"
        
    return "\n".join(content_lines)

def process_directory_for_codepen(texts_dir, copy_clipboard=True):
    """Process all text files in the given directory for CodePen application."""
    # Dictionary to store all processed files
    json_data = {}
    counter = 1
    
    # Check if input directory exists
    if not os.path.exists(texts_dir):
        print(f"Error: Input directory '{texts_dir}' does not exist.")
        return json_data
    
    # Get all items in the folder
    try:
        items = os.listdir(texts_dir)
    except PermissionError:
        print(f"Permission denied: {texts_dir}")
        return json_data
    
    # Process all .txt files
    txt_files = [f for f in items if f.endswith('.txt') and os.path.isfile(os.path.join(texts_dir, f))]
    for file_name in sorted(txt_files):
        file_path = os.path.join(texts_dir, file_name)
        title = extract_title_from_file(file_path)
        content = process_txt_file(file_path, texts_dir)
        
        if content:  # Only add non-empty content
            json_data[str(counter)] = {
                "title": title,
                "content": content
            }
            counter += 1
    
    # Process all .csv files
    csv_files = [f for f in items if f.endswith('.csv') and os.path.isfile(os.path.join(texts_dir, f))]
    for file_name in sorted(csv_files):
        file_path = os.path.join(texts_dir, file_name)
        title = extract_title_from_file(file_path)
        content = process_csv_file(file_path)
        
        if content:  # Only add non-empty content
            json_data[str(counter)] = {
                "title": title,
                "content": content
            }
            counter += 1
    
    # Create JavaScript code with template data
    js_data_content = f"const templates = {json.dumps(json_data, ensure_ascii=False, indent=2)};\n\n"
    
    # Read script_without_template_data_base.js content
    script_js_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "script_without_template_data_base.js")
    script_js_content = ""
    
    if os.path.exists(script_js_path):
        with open(script_js_path, 'r', encoding='utf-8') as f:
            script_js_content = f.read()
        print(f"✅ Loaded script_without_template_data_base.js content ({len(script_js_content)} characters)")
    else:
        print(f"❌ script_without_template_data_base.js not found at {script_js_path}")
        return json_data
    
    # Combine data and script content
    combined_content = js_data_content + script_js_content
    
    # Save combined content to script.js for local testing
    script_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "script.js")
    with open(script_output_path, 'w', encoding='utf-8') as f:
        f.write(combined_content)
    print(f"✅ Combined content saved to {script_output_path}")
    
    if copy_clipboard:
        copy_to_clipboard(combined_content)
        print(f"📋 Combined content copied to clipboard (template data + script_without_template_data_base.js)")
    
    print(f"Generated template data with {len(json_data)} entries")
    print("Files processed:")
    for key, data in json_data.items():
        print(f"  - {data['title']}")
    
    print(f"\n🚀 Ready for CodePen! Paste the clipboard content into CodePen's JS section.")
    print("📖 Make sure to use the HTML from index.html in this directory")
    
    return json_data

def process_directory(texts_dir, output_path=None):
    """Process all text files in the given directory (original function)."""
    # Dictionary to store all processed files
    json_data = {}
    counter = 1
    
    # Process all .txt files
    txt_files = [f for f in os.listdir(texts_dir) if f.endswith('.txt')]
    for file_name in sorted(txt_files):
        file_path = os.path.join(texts_dir, file_name)
        title = extract_title_from_file(file_path)
        content = process_txt_file(file_path, texts_dir)
        
        if content:  # Only add non-empty content
            json_data[str(counter)] = {
                "title": title,
                "content": content
            }
            counter += 1
    
    # Process all .csv files
    csv_files = [f for f in os.listdir(texts_dir) if f.endswith('.csv')]
    for file_name in sorted(csv_files):
        file_path = os.path.join(texts_dir, file_name)
        title = extract_title_from_file(file_path)
        content = process_csv_file(file_path)
        
        if content:  # Only add non-empty content
            json_data[str(counter)] = {
                "title": title,
                "content": content
            }
            counter += 1
    
    # Output the JSON data
    output_js = f"const templates = {json.dumps(json_data, ensure_ascii=False, indent=2)}"
    
    # Determine output path
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "template_data.js")
    
    # Save to a JS file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_js)
    
    print(f"Generated template data with {len(json_data)} entries to {output_path}")
    
    return json_data

def copy_to_clipboard(content):
    """Copy the contents to clipboard."""
    try:
        # For macOS
        if sys.platform == "darwin":
            subprocess.run(['pbcopy'], input=content, text=True, check=True)
            print(f"✅ Content copied to clipboard")
            
        # For Linux with xclip
        elif sys.platform.startswith('linux'):
            subprocess.run(['xclip', '-selection', 'clipboard'], input=content, text=True, check=True)
            print(f"✅ Content copied to clipboard")
            
        # For Windows with clip
        elif sys.platform == "win32":
            subprocess.run(['clip'], input=content, text=True, check=True)
            print(f"✅ Content copied to clipboard")
            
    except subprocess.CalledProcessError:
        print(f"❌ Failed to copy to clipboard")
    except FileNotFoundError:
        print(f"❌ Clipboard utility not found")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process text files and generate template data.')
    parser.add_argument('-p', '--path', default='./folder_w_text',
                        help='Directory containing the text files (default: ./folder_w_text)')
    parser.add_argument('-i', '--input', dest='path',
                        help='Alias for -p/--path (for backwards compatibility)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output file path (default: template_data.js)')
    parser.add_argument('--no-copy', action='store_true',
                        help='Skip copying to clipboard')

    args = parser.parse_args()

    # If no output specified, ask user
    if args.output is None:
        print(f"Processing files from: {args.path}")
        print("To change input folder, use: python process_texts.py -p ./folder_name")
        print("***************WARNING::: MAKE SURE TO COPY OVER AND PASTE OVER script_without_template_data_base.js")
        output_file = input("Output file (blank for clipboard only, 'script.js' to overwrite script.js): ").strip()
        if not output_file:
            # Just copy to clipboard, no file output
            process_directory_for_codepen(args.path, copy_clipboard=not args.no_copy)
            return
        else:
            args.output = output_file

    # Check if user wants to generate script.js (combined file)
    if args.output == "script.js":
        # Generate combined script.js
        process_directory_for_codepen(args.path, copy_clipboard=not args.no_copy)
    else:
        # Generate specified output file
        process_directory(args.path, args.output)
        
        # Copy to clipboard unless --no-copy flag is used
        if not args.no_copy:
            with open(args.output, 'r', encoding='utf-8') as f:
                file_content = f.read()
            copy_to_clipboard(file_content)

if __name__ == "__main__":
    main()