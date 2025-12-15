# Shakespeare Text to JSON Converter & Viewer

This project converts Shakespeare play text files into structured JSON format and provides an interactive HTML viewer for reading and searching the plays.

## Files

- **`text_to_json.py`** - Python script to convert text files to JSON
- **`romeo_juliet.txt`** - Source text file (Project Gutenberg format)
- **`romeo_juliet.json`** - Generated JSON file with structured play data
- **`shakespeare_viewer.html`** - Interactive HTML viewer for the play
- **`grabshakespeare.py`** - Original extraction script (legacy)

## Quick Start

### 1. Convert Text to JSON

```bash
python text_to_json.py romeo_juliet.txt romeo_juliet.json
```

**Output:**
```
Reading romeo_juliet.txt...
Writing JSON to romeo_juliet.json...

Conversion complete!
Total Acts: 5
Total Scenes: 24
Total Lines/Entries: 1020
Output saved to: romeo_juliet.json
```

### 2. View in Browser

Simply open `shakespeare_viewer.html` in your web browser. The viewer will automatically load `romeo_juliet.json` from the same directory.

## JSON Structure

The generated JSON follows this structure:

```json
{
  "title": "Romeo and Juliet",
  "author": "William Shakespeare",
  "acts": [
    {
      "act_number": 1,
      "act_title": "ACT I",
      "scenes": [
        {
          "scene_number": 1,
          "scene_title": "Scene I. A public place.",
          "location": "A PUBLIC PLACE.",
          "lines": [
            {
              "speaker": "STAGE_DIRECTION",
              "text": "Enter Sampson and Gregory..."
            },
            {
              "speaker": "SAMPSON",
              "text": "Gregory, on my word, we'll not carry coals."
            }
          ]
        }
      ]
    }
  ]
}
```

## HTML Viewer Features

### Navigation
- **Act/Scene Selection**: Dropdown menus to browse by act and scene
- **Display Scene**: View complete scenes with formatted dialogue and stage directions

### Search Functions
- **Search Play**: Find text across the entire play and display complete scenes
  - Shows the **entire scene** containing the search results (not just the matching line)
  - Highlights matching text in yellow with bold formatting
  - Highlights matching lines with yellow background
  - Shows count of matches and number of scenes found
  - Perfect for understanding context around any quote or phrase
- **Extract Lines**: Extract a specified number of lines starting from any text
  - Automatically copies extracted text to clipboard
  - Useful for creating study excerpts or quotations

### Additional Features
- **Random Scene**: Jump to a random scene for exploration
- **Responsive Design**: Works on desktop and mobile devices
- **Formatted Output**:
  - Stage directions in italics
  - Character names in bold
  - Proper spacing and typography

## Converting Other Shakespeare Plays

To convert another play:

1. Download the text file from [Project Gutenberg](https://www.gutenberg.org/)
2. Run the conversion script:
   ```bash
   python text_to_json.py hamlet.txt hamlet.json
   ```
3. Update the `shakespeare_viewer.html` to load your JSON file:
   ```javascript
   // In shakespeare_viewer.html, find this line:
   const response = await fetch('romeo_juliet.json');
   // Change to:
   const response = await fetch('hamlet.json');
   ```

## Script Details

### text_to_json.py

**Features:**
- Parses Act and Scene structure (Roman numerals I-V)
- Identifies character dialogue (names ending with periods)
- Captures stage directions (Enter, Exit, Exeunt, etc.)
- Handles scene locations and descriptions
- Skips table of contents and metadata

**Requirements:**
- Python 3.6+
- No external dependencies (uses only standard library)

**Usage:**
```bash
python text_to_json.py <input_file> [output_file]
```

If `output_file` is not specified, it defaults to the input filename with `.json` extension.

## Legacy Tool: grabshakespeare.py

The original `grabshakespeare.py` script extracts text segments from the play:

```bash
# By Act/Scene reference
python grabshakespeare.py romeo_juliet.txt "Act 1 Scene 1"

# By searching for text
python grabshakespeare.py romeo_juliet.txt "Two households"

# With custom line count
python grabshakespeare.py romeo_juliet.txt "Act 2" --lines=300
```

**Features:**
- Extracts 500 lines by default (configurable)
- Finds text by Act/Scene or by content search
- Copies extracted text to clipboard (requires `pyperclip`)
- Shows preview of extracted content

## Comparison with Bible JSON Loader

This system is modeled after the `json_loader/json_loader.html` which loads Bible JSON data. Key similarities:

| Feature | Bible Loader | Shakespeare Viewer |
|---------|-------------|-------------------|
| Data Format | JSON | JSON |
| Structure | Books/Chapters/Verses | Acts/Scenes/Lines |
| Search | Text search | Text search + extraction |
| Navigation | Dropdown menus | Dropdown menus |
| Display | Formatted verses | Formatted dialogue |

## Tips & Tricks

### Searching for Text

1. Enter any text in "Search for text" field
2. Click "Search Play"
3. The complete scene(s) containing that text will be displayed
4. Matching text is highlighted in yellow
5. Matching lines have a yellow background for easy spotting

**Example:** Searching for "what, shall this speech" will display:
- The complete Act I, Scene IV
- All dialogue and stage directions in that scene
- The matching line highlighted with a yellow background
- The search text itself marked in bold yellow

### Extracting Specific Passages

1. Enter the starting text in "Search for text"
2. Set "Number of lines to extract" (default: 500)
3. Click "Extract Lines"
4. Text is automatically copied to clipboard
5. Continues across multiple scenes if needed

### Study Usage

The viewer is perfect for:
- Reading the play scene by scene
- Finding famous quotes
- Creating study guides or excerpts
- Analyzing character dialogue
- Exploring the play's structure

### Performance

- JSON file size: ~222KB for Romeo and Juliet
- Loads instantly in modern browsers
- Handles searches across all 5 acts and 24 scenes efficiently
- No external dependencies or internet connection required

## Troubleshooting

**Problem:** HTML viewer shows "Failed to load play data"
- **Solution:** Make sure `romeo_juliet.json` is in the same directory as `shakespeare_viewer.html`

**Problem:** Conversion script creates empty scenes
- **Solution:** Check that the source text follows Project Gutenberg format with clear Act/Scene markers

**Problem:** Characters names are misidentified
- **Solution:** The script looks for names ending with periods (e.g., "ROMEO."). Verify your source text follows this format.

## Future Enhancements

Possible improvements:
- [ ] Support for other Shakespeare plays (automatic play detection)
- [ ] Character-specific filtering (show all lines by Romeo)
- [ ] Line numbering
- [ ] Export to PDF or other formats
- [ ] Side-by-side comparison with modern translations
- [ ] Audio playback (text-to-speech)

## Credits

- Text source: [Project Gutenberg](https://www.gutenberg.org/)
- Inspired by the Bible JSON loader system
- Created for educational and literary analysis purposes

## License

This tool is provided for educational purposes. The Shakespeare texts are in the public domain.
