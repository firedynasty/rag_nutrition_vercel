#!/usr/bin/env python3
"""
Replace specific lines in a file with clipboard content.

Usage:
    python replace_lines.py <start_line> <end_line> [filename]

Examples:
    python replace_lines.py 421 444
    python replace_lines.py 421 444 romeo_and_juliet_no_fear.txt

The script will:
1. Read content from your clipboard
2. Auto-balance lines if clipboard has different number of lines
3. Replace lines <start_line> through <end_line> (inclusive) with clipboard content
4. Create a backup of the original file (.bak)
5. Show you what changed

  - ✨ Auto-balances lines (merges/pads)


"""

import sys
import os
import re
import pyperclip
from pathlib import Path

def smart_split_lines(text, target_count):
    """
    Split text into approximately target_count lines intelligently.
    Tries to break at sentence boundaries (. ! ?) first, then at commas.
    """
    # Clean up the text
    text = text.strip()

    # Split into sentences (at . ! ?)
    sentences = re.split(r'([.!?]\s+)', text)

    # Rejoin sentence with its punctuation
    parts = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i+1].strip() in ['.', '!', '?', '. ', '! ', '? ']:
            parts.append(sentences[i] + sentences[i+1])
            i += 2
        else:
            if sentences[i].strip():
                parts.append(sentences[i])
            i += 1

    # If we don't have enough parts, split by commas
    if len(parts) < target_count:
        new_parts = []
        for part in parts:
            if ',' in part:
                # Split by comma but keep the comma
                comma_parts = part.split(',')
                for j, cp in enumerate(comma_parts[:-1]):
                    new_parts.append(cp + ',')
                if comma_parts[-1].strip():
                    new_parts.append(comma_parts[-1])
            else:
                new_parts.append(part)
        parts = new_parts

    # If still not enough, split long parts by word count
    if len(parts) < target_count:
        new_parts = []
        for part in parts:
            words = part.split()
            if len(words) > 15:  # If part has more than 15 words, split it
                mid = len(words) // 2
                new_parts.append(' '.join(words[:mid]))
                new_parts.append(' '.join(words[mid:]))
            else:
                new_parts.append(part)
        parts = new_parts

    # If we have too many parts, merge them
    while len(parts) > target_count:
        # Find the two shortest adjacent parts to merge
        min_idx = 0
        min_len = len(parts[0]) + len(parts[1])
        for i in range(len(parts) - 1):
            combined_len = len(parts[i]) + len(parts[i+1])
            if combined_len < min_len:
                min_len = combined_len
                min_idx = i

        # Merge the two parts
        parts[min_idx] = parts[min_idx] + ' ' + parts[min_idx + 1]
        parts.pop(min_idx + 1)

    # Clean up and ensure each line ends with newline
    lines = [part.strip() + '\n' for part in parts if part.strip()]

    return lines

def smart_merge_lines(lines, target_count):
    """
    Merge lines down to target_count by combining adjacent lines.
    """
    while len(lines) > target_count:
        # Find the two shortest adjacent lines to merge
        min_idx = 0
        min_len = len(lines[0].strip()) + len(lines[1].strip())

        for i in range(len(lines) - 1):
            combined_len = len(lines[i].strip()) + len(lines[i+1].strip())
            if combined_len < min_len:
                min_len = combined_len
                min_idx = i

        # Merge the two lines
        merged = lines[min_idx].strip() + ' ' + lines[min_idx + 1].strip() + '\n'
        lines[min_idx] = merged
        lines.pop(min_idx + 1)

    return lines

def balance_lines(clipboard_lines, target_count):
    """
    Balance clipboard lines to match target_count.
    """
    current_count = len(clipboard_lines)

    if current_count == target_count:
        return clipboard_lines, "exact match"

    # Combine all text
    full_text = ' '.join(line.strip() for line in clipboard_lines)

    if current_count > target_count:
        # Too many lines - merge them
        print(f"⚠️  Clipboard has {current_count} lines, but target is {target_count}")
        print(f"   🔀 Merging lines to fit...")
        result = smart_merge_lines(clipboard_lines[:], target_count)
        return result, "merged"
    else:
        # Too few lines - add filler lines
        print(f"⚠️  Clipboard has {current_count} lines, but target is {target_count}")
        print(f"   ➕ Adding {target_count - current_count} filler line(s)...")
        result = clipboard_lines[:]
        filler_line = "_\n"
        for _ in range(target_count - current_count):
            result.append(filler_line)
        return result, "padded"

def replace_lines(filename, start_line, end_line, clipboard_content, auto_balance=True):
    """
    Replace lines start_line through end_line with clipboard content.
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

    # Balance lines if needed and auto_balance is enabled
    balance_action = "none"
    if auto_balance and len(new_lines) != target_count:
        new_lines, balance_action = balance_lines(new_lines, target_count)

    # Show summary
    print(f"\n📋 Replacing lines {start_line}-{end_line} ({target_count} lines)")
    print(f"   with {len(new_lines)} lines from clipboard")
    if balance_action != "none":
        print(f"   ✨ Lines were {balance_action} to match target count")

    # Show context before the replacement section
    print("📍 CONTEXT (3 lines before):")
    print("─" * 80)
    context_start = max(0, start_idx - 3)
    for i in range(context_start, start_idx):
        print(f"   {i+1:4d} │ {lines[i]}", end='')
    print("─" * 80)

    print("\n🗑️  OLD CONTENT (will be replaced):")
    print("─" * 80)
    for i in range(start_idx, end_idx):
        print(f"   {i+1:4d} │ {lines[i]}", end='')
    print("─" * 80)

    print("\n📍 CONTEXT (3 lines after):")
    print("─" * 80)
    context_end = min(len(lines), end_idx + 3)
    for i in range(end_idx, context_end):
        print(f"   {i+1:4d} │ {lines[i]}", end='')
    print("─" * 80)

    print("\n✨ NEW CONTENT:")
    print("─" * 80)
    for i, line in enumerate(new_lines, start=start_line):
        print(f"   {i:4d} │ {line}", end='')
    print("─" * 80)
    
    # Warn if line counts don't match
    if len(new_lines) != target_count:
        print(f"\n⚠️  WARNING: Line count mismatch!")
        print(f"   Expected: {target_count} lines")
        print(f"   Got: {len(new_lines)} lines")
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
        print("Usage: python replace_lines.py <start_line> <end_line> [filename]")
        print(f"\nDefault filename: {default_filename}")
        print("\nExamples:")
        print("  python replace_lines.py 421 444")
        print("  python replace_lines.py 421 444 romeo_and_juliet_no_fear.txt")
        print("\nFeatures:")
        print("  ✂️  Auto-balances lines to match target count")
        print("  🔀 Intelligently merges or splits text")
        print("  🔒 Creates backup before changes")
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

    # Perform the replacement with auto-balancing
    success = replace_lines(filename, start_line, end_line, clipboard_content, auto_balance=True)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
