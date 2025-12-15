#!/usr/bin/env python3
"""
Convert Shakespeare play text file to structured JSON format.
Handles "Act X Scene Y" format (like tempest1.txt).
Usage: python text_to_json_v2.py input.txt output.json
"""

import sys
import json
import re
from typing import List, Dict, Any


def parse_shakespeare_play(filename: str) -> Dict[str, Any]:
    """
    Parse Shakespeare play text file and create structured JSON.
    Handles "Act X Scene Y" format.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract title from first lines
    title = "Unknown Play"
    author = "William Shakespeare"
    for line in lines[:50]:
        stripped = line.strip().upper()
        if 'TEMPEST' in stripped and not stripped.startswith('SCENE'):
            title = "The Tempest"
            break
        elif 'ROMEO' in stripped and 'JULIET' in stripped:
            title = "Romeo and Juliet"
            break

    play_data = {
        "title": title,
        "author": author,
        "acts": []
    }

    current_act = None
    current_scene = None
    current_speaker = None
    line_buffer = []
    play_started = False

    for line_num, line in enumerate(lines):
        stripped = line.strip()

        # Check for "Act X Scene Y" marker
        act_scene_match = re.match(r'^Act\s+(\d+)\s+Scene\s+(\d+)\s*$', stripped, re.IGNORECASE)
        if act_scene_match:
            play_started = True
            act_num = int(act_scene_match.group(1))
            scene_num = int(act_scene_match.group(2))

            # Save previous speaker's lines
            if current_scene and current_speaker and line_buffer:
                current_scene['lines'].append({
                    'speaker': current_speaker,
                    'text': '\n'.join(line_buffer)
                })
                line_buffer = []
                current_speaker = None

            # Check if we need to create a new act
            if current_act is None or current_act['act_number'] != act_num:
                current_act = {
                    "act_number": act_num,
                    "act_title": f"ACT {to_roman(act_num)}",
                    "scenes": []
                }
                play_data["acts"].append(current_act)

            # Look ahead for scene location (usually on previous or next lines)
            scene_location = ""
            if line_num > 0:
                prev_line = lines[line_num - 1].strip()
                if prev_line and not re.match(r'^Act\s+\d+', prev_line):
                    scene_location = prev_line

            if not scene_location and line_num + 1 < len(lines):
                next_line = lines[line_num + 1].strip()
                if next_line and not re.match(r'^(Enter|Exit)', next_line, re.IGNORECASE):
                    scene_location = next_line

            current_scene = {
                "scene_number": scene_num,
                "scene_title": f"Scene {to_roman(scene_num)}. {scene_location}" if scene_location else f"Scene {to_roman(scene_num)}",
                "location": scene_location.upper() if scene_location else "",
                "lines": []
            }
            current_act["scenes"].append(current_scene)
            continue

        if not play_started:
            continue

        # Check for stage directions (Enter, Exit, Exeunt, etc.)
        if re.match(r'^(Enter|Exit|Exeunt|Re-enter|Aside|Within)(\s+|$)', stripped, re.IGNORECASE):
            # Save previous speaker's lines
            if current_speaker and line_buffer:
                current_scene['lines'].append({
                    'speaker': current_speaker,
                    'text': '\n'.join(line_buffer)
                })
                line_buffer = []
                current_speaker = None

            current_scene['lines'].append({
                'speaker': 'STAGE_DIRECTION',
                'text': stripped
            })
            continue

        # Check if this line is a speaker name
        # Speaker names are: all caps, on their own line, short, not stage directions
        if stripped and stripped.isupper() and len(stripped) < 30:
            # Make sure it's not a stage direction
            if not re.match(r'^(Enter|Exit|Exeunt|Re-enter|Aside|Within|Alarum|Flourish)\s*', stripped, re.IGNORECASE):
                # Save previous speaker's lines
                if current_speaker and line_buffer:
                    current_scene['lines'].append({
                        'speaker': current_speaker,
                        'text': '\n'.join(line_buffer)
                    })
                    line_buffer = []

                current_speaker = stripped
                continue

        # Regular dialogue line
        if stripped and current_speaker:
            line_buffer.append(stripped)

    # Save final buffered lines
    if current_scene and current_speaker and line_buffer:
        current_scene['lines'].append({
            'speaker': current_speaker,
            'text': '\n'.join(line_buffer)
        })

    return play_data


def to_roman(num: int) -> str:
    """Convert integer to Roman numeral."""
    val = [10, 9, 5, 4, 1]
    syms = ['X', 'IX', 'V', 'IV', 'I']
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num


def main():
    if len(sys.argv) < 2:
        print("Usage: python text_to_json_v2.py input.txt [output.json] [\"Play Title\"]")
        print("Example: python text_to_json_v2.py much_ado.txt much_ado.json \"Much Ado About Nothing\"")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.txt', '.json')

    # Get title from command line or prompt user
    if len(sys.argv) > 3:
        title = sys.argv[3]
    else:
        title = input("Enter the play title (or press Enter for 'Unknown Play'): ").strip()
        if not title:
            title = "Unknown Play"

    try:
        print(f"Reading {input_file}...")
        play_data = parse_shakespeare_play(input_file)

        # Override the title with user input
        play_data['title'] = title

        # Write JSON output
        print(f"Writing JSON to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(play_data, f, indent=2, ensure_ascii=False)

        # Print statistics
        total_acts = len(play_data['acts'])
        total_scenes = sum(len(act['scenes']) for act in play_data['acts'])
        total_lines = sum(
            len(scene['lines'])
            for act in play_data['acts']
            for scene in act['scenes']
        )

        print(f"\nConversion complete!")
        print(f"Play: {play_data['title']}")
        print(f"Total Acts: {total_acts}")
        print(f"Total Scenes: {total_scenes}")
        print(f"Total Lines/Entries: {total_lines}")
        print(f"Output saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
