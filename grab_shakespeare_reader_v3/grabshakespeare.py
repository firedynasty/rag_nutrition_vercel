#!/usr/bin/env python3
"""
Extract Romeo and Juliet text segments starting from a specified location.
Usage: python extract_romeo_juliet.py romeo_juliet.txt "Act 1 Scene 1"
       python extract_romeo_juliet.py romeo_juliet.txt "And too soon marr'd"
"""

import sys
import re
from typing import List, Tuple, Optional

def parse_act_scene_reference(reference: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse an Act/Scene reference like "Act 1 Scene 1" or "Act 2"
    Returns (act, scene) where scene is None if not specified
    """
    reference = reference.strip()
    
    # Match "Act X Scene Y" or "Act X"
    match = re.match(r'act\s+(\d+)(?:\s+scene\s+(\d+))?', reference, re.IGNORECASE)
    
    if match:
        act = int(match.group(1))
        scene = int(match.group(2)) if match.group(2) else None
        return act, scene
    
    return None, None

def read_play_lines(filename: str) -> List[str]:
    """Read all lines from the play file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()

def find_act_scene_line(lines: List[str], act: int, scene: Optional[int] = None) -> int:
    """
    Find the line number where the specified Act/Scene begins.
    Returns the index of the line.
    """
    for i, line in enumerate(lines):
        line_stripped = line.strip().upper()
        
        if scene is not None:
            # Look for "ACT X" (Roman or Arabic numeral)
            act_patterns = [
                f"ACT {act}",
                f"ACT {int_to_roman(act)}",
            ]
            
            if any(pattern in line_stripped for pattern in act_patterns):
                # Check this line and next few lines for scene
                for j in range(i, min(i + 10, len(lines))):
                    check_line = lines[j].strip().upper()
                    scene_patterns = [
                        f"SCENE {scene}",
                        f"SCENE {int_to_roman(scene)}",
                    ]
                    if any(pattern in check_line for pattern in scene_patterns):
                        return j
        else:
            # Just look for Act
            act_patterns = [
                f"ACT {act}",
                f"ACT {int_to_roman(act)}",
            ]
            if any(pattern in line_stripped for pattern in act_patterns):
                return i
    
    raise ValueError(f"Could not find Act {act}{f' Scene {scene}' if scene else ''}")

def find_text_and_context(lines: List[str], search_text: str) -> Tuple[int, str]:
    """
    Find the line containing the search text and identify which Act/Scene it's in.
    Returns (line_index, context_string)
    """
    search_text_lower = search_text.lower().strip()
    
    # Find the line with the text
    found_line = -1
    for i, line in enumerate(lines):
        if search_text_lower in line.lower():
            found_line = i
            break
    
    if found_line == -1:
        raise ValueError(f"Could not find text: '{search_text}'")
    
    # Search backwards for Act/Scene markers
    act_num = None
    scene_num = None
    
    for i in range(found_line, -1, -1):
        line_upper = lines[i].strip().upper()
        
        # Look for Scene first
        scene_match = re.search(r'SCENE\s+([IVXLCDM]+|\d+)', line_upper)
        if scene_match and scene_num is None:
            scene_str = scene_match.group(1)
            # Convert Roman numerals or use digit
            scene_num = roman_to_int(scene_str) if scene_str.isalpha() else int(scene_str)
        
        # Look for Act
        act_match = re.search(r'ACT\s+([IVXLCDM]+|\d+)', line_upper)
        if act_match and act_num is None:
            act_str = act_match.group(1)
            act_num = roman_to_int(act_str) if act_str.isalpha() else int(act_str)
        
        # If we found both, we're done
        if act_num is not None and scene_num is not None:
            break
    
    # Format the context
    if act_num and scene_num:
        context = f"Act {act_num}, Scene {scene_num}"
    elif act_num:
        context = f"Act {act_num}"
    else:
        context = "Unknown location"
    
    return found_line, context

def roman_to_int(s: str) -> int:
    """Convert Roman numeral to integer."""
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev_value = 0
    
    for char in reversed(s.upper()):
        value = roman_values.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    
    return total

def int_to_roman(num: int) -> str:
    """Convert integer to Roman numeral."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num

def extract_lines(lines: List[str], start_idx: int, num_lines: int = 500) -> List[str]:
    """Extract lines starting from start_idx."""
    end_idx = min(start_idx + num_lines, len(lines))
    return lines[start_idx:end_idx]

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_romeo_juliet.py romeo_juliet.txt \"Act 1 Scene 1\"")
        print("       python extract_romeo_juliet.py romeo_juliet.txt \"And too soon marr'd\"")
        sys.exit(1)
    
    filename = sys.argv[1]
    reference = sys.argv[2]
    
    # Allow custom number of lines
    num_lines = 500
    for arg in sys.argv[3:]:
        if arg.startswith("--lines="):
            try:
                num_lines = int(arg.split("=")[1])
            except:
                pass
    
    try:
        # Read the play
        print(f"Reading {filename}...")
        lines = read_play_lines(filename)
        print(f"Total lines in file: {len(lines)}")
        
        # Try to parse as Act/Scene reference
        act, scene = parse_act_scene_reference(reference)
        
        if act is not None:
            # It's an Act/Scene reference
            start_idx = find_act_scene_line(lines, act, scene)
            location = f"Act {act}{f', Scene {scene}' if scene else ''}"
            print(f"\nFound {location} at line {start_idx + 1}")
        else:
            # It's a text search - find the context
            start_idx, context = find_text_and_context(lines, reference)
            print(f"\nFound text at line {start_idx + 1}")
            print(f"Location: {context}")
            location = context
        
        # Extract the segment
        segment = extract_lines(lines, start_idx, num_lines)
        
        print(f"\nExtracting {len(segment)} lines starting from line {start_idx + 1}")
        print("=" * 60)
        
        # Show preview (first 20 lines)
        print(f"\n[Preview - First 20 lines from {location}]\n")
        for i, line in enumerate(segment[:20]):
            print(line.rstrip())
        
        if len(segment) > 20:
            print(f"\n... and {len(segment) - 20} more lines ...")
        
        print("\n" + "=" * 60)
        
        # Combine all text
        combined_text = "".join(segment)
        word_count = len(combined_text.split())
        print(f"Total words: {word_count}")
        
        # Copy to clipboard
        try:
            import pyperclip
            pyperclip.copy(combined_text)
            print("\n[Text copied to clipboard]")
        except ImportError:
            print("\n[pyperclip not available - install with: pip install pyperclip]")
            print("\nFirst 500 characters:")
            print(combined_text[:500] + "..." if len(combined_text) > 500 else combined_text)
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
