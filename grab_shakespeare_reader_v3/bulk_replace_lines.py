#!/usr/bin/env python3
"""
Bulk replace lines in a file with clipboard content (no auto-balancing).

Usage:
    python bulk_replace_lines.py <start_line> <end_line> [filename]

Examples:
    python bulk_replace_lines.py 1780 1870
    python bulk_replace_lines.py 421 444 romeo_and_juliet_modern.txt

The script will:
1. Read content from your clipboard
2. Replace lines exactly as-is (NO auto-balancing)
3. Create a backup of the original file (.bak)
4. Show summary and ask for confirmation
"""

import sys
import os
import pyperclip
from pathlib import Path

def bulk_replace_lines(filename, start_line, end_line, clipboard_content):
    """
    Replace lines start_line through end_line with clipboard content.
    NO auto-balancing - uses clipboard content exactly as-is.
    Line numbers are 1-indexed (like text editors).
    """
    # Convert to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line  # This will be the line AFTER the last line to replace

    # Read the file
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

    # Validate line numbers
    if start_idx < 0 or start_idx >= len(lines):
        print(f"❌ Error: Start line {start_line} is out of range (file has {len(lines)} lines)")
        return False

    if end_idx < start_idx or end_idx > len(lines):
        print(f"❌ Error: End line {end_line} is out of range (file has {len(lines)} lines)")
        return False

    # Create backup
    backup_file = filename + '.bak'
    try:
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ Backup created: {backup_file}")
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False

    # Split clipboard content into lines
    new_lines = clipboard_content.splitlines(keepends=True)

    # If the last line doesn't have a newline, add one
    if new_lines and not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'

    # Calculate target line count
    target_count = end_line - start_line + 1

    # Show summary
    print(f"\n📋 Replacing lines {start_line}-{end_line} ({target_count} lines)")
    print(f"   with {len(new_lines)} lines from clipboard (NO auto-balancing)")

    # Warn if line counts don't match
    if len(new_lines) != target_count:
        print(f"\n⚠️  WARNING: Line count mismatch!")
        print(f"   Target range: {target_count} lines")
        print(f"   Clipboard has: {len(new_lines)} lines")
        print(f"   Difference: {len(new_lines) - target_count:+d} lines")
        print(f"   This will shift all subsequent line numbers!")

    # Ask for confirmation
    response = input("\n❓ Proceed with replacement? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Cancelled")
        return False

    # Replace the lines
    new_file_lines = lines[:start_idx] + new_lines + lines[end_idx:]

    # Write back to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(new_file_lines)
        print(f"\n✅ Successfully replaced lines {start_line}-{end_line} in '{filename}'")
        print(f"   Original file backed up to '{backup_file}'")
        print(f"   File now has {len(new_file_lines)} lines (was {len(lines)} lines)")
        return True
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        print(f"   Your original file is safe in '{backup_file}'")
        return False

def main():
    # Default filename
    default_filename = "romeo_and_juliet_modern.txt"

    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python bulk_replace_lines.py <start_line> <end_line> [filename]")
        print(f"\nDefault filename: {default_filename}")
        print("\nExamples:")
        print("  python bulk_replace_lines.py 1780 1870")
        print("  python bulk_replace_lines.py 421 444 romeo_and_juliet_modern.txt")
        print("\nFeatures:")
        print("  📋 Replaces lines exactly as-is (NO auto-balancing)")
        print("  🔒 Creates backup before changes")
        print("  ⚡ Fast - no preview, just summary")
        sys.exit(1)

    try:
        start_line = int(sys.argv[1])
        end_line = int(sys.argv[2])
    except ValueError:
        print("❌ Error: Start and end line must be integers")
        sys.exit(1)

    # Get filename from args or use default
    if len(sys.argv) >= 4:
        filename = sys.argv[3]
    else:
        filename = default_filename
        print(f"📄 Using default file: {filename}\n")

    # Validate line range
    if start_line <= 0 or end_line <= 0:
        print("❌ Error: Line numbers must be positive")
        sys.exit(1)

    if start_line > end_line:
        print("❌ Error: Start line must be less than or equal to end line")
        sys.exit(1)

    # Get clipboard content
    try:
        clipboard_content = pyperclip.paste()
    except Exception as e:
        print(f"❌ Error accessing clipboard: {e}")
        print("\n💡 Make sure you have pyperclip installed:")
        print("   pip install pyperclip")
        sys.exit(1)

    if not clipboard_content:
        print("❌ Error: Clipboard is empty")
        sys.exit(1)

    print(f"📋 Clipboard contains {len(clipboard_content)} characters")
    print(f"   ({len(clipboard_content.splitlines())} lines)\n")

    # Perform the replacement (NO auto-balancing)
    success = bulk_replace_lines(filename, start_line, end_line, clipboard_content)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
