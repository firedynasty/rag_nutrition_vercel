#!/usr/bin/env python3
"""
Convert Romeo and Juliet text file to structured JSON format.
Usage: python text_to_json.py romeo_juliet.txt output.json
"""

import sys
import json
import re
from typing import List, Dict, Any


def parse_romeo_juliet(filename: str) -> Dict[str, Any]:
    """
    Parse Romeo and Juliet text file and create structured JSON.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    play_data = {
        "title": "Romeo and Juliet",
        "author": "William Shakespeare",
        "acts": []
    }

    current_act = None
    current_scene = None
    current_speaker = None
    line_buffer = []

    # Track if we've started the actual play content
    play_started = False
    in_scene_header = False
    entered_actual_content = False

    for line_num, line in enumerate(lines):
        stripped = line.strip()
        upper = stripped.upper()

        # Look for stage directions like "Enter" which mark actual play content
        # Handle both regular and italic formats (_Enter_)
        if not entered_actual_content:
            if re.search(r'(^|_)\s*Enter\s+', stripped, re.IGNORECASE):
                entered_actual_content = True

        # Skip until we find the first ACT after we've entered actual content
        if not play_started:
            if entered_actual_content and re.match(r'^ACT\s+[IVX]+\.?\s*$', upper):
                play_started = True
            else:
                continue

        # Check for ACT marker (with or without period)
        act_match = re.match(r'^ACT\s+([IVX]+)\.?\s*$', upper)
        if act_match:
            # Save previous scene if exists
            if current_scene and line_buffer:
                current_scene['lines'].append({
                    'speaker': current_speaker,
                    'text': '\n'.join(line_buffer)
                })
                line_buffer = []

            # Create new act
            act_num = roman_to_int(act_match.group(1))
            current_act = {
                "act_number": act_num,
                "act_title": stripped,
                "scenes": []
            }
            play_data["acts"].append(current_act)
            current_scene = None
            current_speaker = None
            in_scene_header = True
            continue

        # Check for SCENE marker
        scene_match = re.match(r'^SCENE\s+([IVX]+)\.?\s*(.*)', upper)
        if scene_match and current_act:
            in_scene_header = True
            # Save previous scene's last dialogue
            if current_scene and line_buffer:
                current_scene['lines'].append({
                    'speaker': current_speaker,
                    'text': '\n'.join(line_buffer)
                })
                line_buffer = []

            # Create new scene
            scene_num = roman_to_int(scene_match.group(1))
            scene_location = scene_match.group(2).strip() if scene_match.group(2) else ""

            # Look ahead for location on next line if not on same line
            if not scene_location and line_num + 1 < len(lines):
                next_line = lines[line_num + 1].strip()
                if next_line and not re.match(r'^(ACT|SCENE|[A-Z\s]+\.)', next_line.upper()):
                    scene_location = next_line

            current_scene = {
                "scene_number": scene_num,
                "scene_title": stripped,
                "location": scene_location,
                "lines": []
            }
            current_act["scenes"].append(current_scene)
            current_speaker = None
            continue

        # Skip "Act X Scene Y" repetition lines
        if re.match(r'^Act\s+\d+\s+Scene\s+\d+\s*$', stripped):
            in_scene_header = False
            continue

        # Check if this is a stage direction starting with "Enter", "Exit", etc.
        # Handle both regular and italic formats (_Enter_, [_Exit._])
        if current_scene and re.search(r'(^|_|\s)(Enter|Exit|Exeunt|Re-enter)\s+', stripped, re.IGNORECASE):
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
            in_scene_header = False
            continue

        # Check for speaker (character name ending with a period)
        # Handle both regular format "ROMEO." and italic format "_Rom._"
        speaker_match = re.match(r'^_?([A-Z][A-Za-z\s\'\-]+)\._?\s*$', stripped)
        if speaker_match and current_scene and not in_scene_header:
            # Save previous speaker's lines
            if current_speaker and line_buffer:
                current_scene['lines'].append({
                    'speaker': current_speaker,
                    'text': '\n'.join(line_buffer)
                })
                line_buffer = []

            # New speaker - clean up any underscores
            current_speaker = speaker_match.group(1).strip().upper()
            continue

        # Regular line of dialogue or stage direction
        if stripped and current_scene and not in_scene_header:
            # Stage directions (in brackets or parentheses)
            if stripped.startswith('[') or stripped.startswith('('):
                if current_speaker and line_buffer:
                    current_scene['lines'].append({
                        'speaker': current_speaker,
                        'text': '\n'.join(line_buffer)
                    })
                    line_buffer = []

                current_scene['lines'].append({
                    'speaker': 'STAGE_DIRECTION',
                    'text': stripped
                })
                current_speaker = None
            else:
                # Regular dialogue line
                if current_speaker:
                    line_buffer.append(stripped)
                elif not re.match(r'^(DRAMATIS|CHARACTERS|SCENE\.)', upper):
                    # Could be narrative text or something else
                    # Only add if we're past the headers
                    pass

    # Save final buffered lines
    if current_scene and current_speaker and line_buffer:
        current_scene['lines'].append({
            'speaker': current_speaker,
            'text': '\n'.join(line_buffer)
        })

    return play_data


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python text_to_json.py romeo_juliet.txt [output.json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'romeo_juliet.json'

    try:
        print(f"Reading {input_file}...")
        play_data = parse_romeo_juliet(input_file)

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
