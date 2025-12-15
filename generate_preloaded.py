#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate preloaded reports JSON files for the React app.

This script scans ./preloaded_reports/ for folders containing .txt and .md files,
and generates JSON files in ./public/preloaded/ that the React app can fetch.

Usage:
    python generate_preloaded.py

Output:
    ./public/preloaded/index.json - List of available folders
    ./public/preloaded/<folder_name>.json - Contents of each folder
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
    """Generate JSON files from preloaded report files (depth=1, files at root level)."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    print(f"Processing files in: {input_dir}")

    # Get all .txt and .md files directly in input_dir (depth=1, no subfolders)
    files_data = {}
    entries = os.listdir(input_dir)
    text_files = [f for f in entries if (f.endswith('.txt') or f.endswith('.md')) and os.path.isfile(os.path.join(input_dir, f))]
    text_files.sort(key=lambda x: x.lower())

    file_count = 0
    for filename in text_files:
        file_path = os.path.join(input_dir, filename)
        content = process_file(file_path)

        if content is not None:
            files_data[filename] = content
            file_count += 1
            print(f"  - {filename} ({len(content):,} chars)")

    if file_count > 0:
        # Save all files to a single JSON named "reports"
        reports_json_path = os.path.join(output_dir, "reports.json")
        with open(reports_json_path, 'w', encoding='utf-8') as f:
            json.dump(files_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {reports_json_path}")
    else:
        print(f"  No .txt or .md files found.")

    # Save index with list of filenames
    index_path = os.path.join(output_dir, 'index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({'files': text_files}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Generated index.json with {file_count} file(s)")
    print(f"📁 Output directory: {output_dir}")

    return text_files

def main():
    parser = argparse.ArgumentParser(description='Generate preloaded reports JSON files.')
    parser.add_argument('input_dir', nargs='?', default='./preloaded_reports',
                        help='Input directory containing report files (default: ./preloaded_reports)')
    parser.add_argument('-o', '--output', default='./public/preloaded',
                        help='Output directory for JSON files (default: ./public/preloaded)')

    args = parser.parse_args()

    generate_preloaded_reports(args.input_dir, args.output)

if __name__ == "__main__":
    main()
