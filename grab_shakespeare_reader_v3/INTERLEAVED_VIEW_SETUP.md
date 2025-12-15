# Interleaved View Setup Complete

## Summary

Successfully implemented an interleaved view feature for Romeo and Juliet that displays both original Shakespearean text and modern translation side-by-side.

## What Was Done

### 1. JSON Conversion ✅

**Created two JSON files from text files:**

```bash
python text_to_json_v2.py romeo_and_juliet.txt romeo_and_juliet.json "Romeo and Juliet"
python text_to_json_v2.py romeo_and_juliet_modern.txt romeo_and_juliet_modern.json "Romeo and Juliet (Modern)"
```

**Results:**
- `romeo_and_juliet.json` - Original Shakespearean text
  - 5 Acts, 24 Scenes, 804 Lines/Entries
- `romeo_and_juliet_modern.json` - Modern translation
  - 5 Acts, 24 Scenes, 804 Lines/Entries

### 2. Removed Play Selector ✅

**Changed in `indexRegular.html`:**
- Removed the entire play selector dropdown (lines 499-506)
- Now loads only Romeo and Juliet (both versions)
- Removed `setupPlaySelector()` function call

### 3. Added Interleaved View Toggle ✅

**New Button:**
```html
<button onclick="toggleViewMode()" class="secondary" id="viewModeButton"
  title="Switch view mode">⇅ Interleaved</button>
```

**Three View Modes:**
1. **📜 Original** - Shows only original Shakespearean text
2. **📖 Modern** - Shows only modern translation
3. **⇅ Interleaved** - Shows both alternating (original → modern → original → modern)

### 4. Added Visual Styling ✅

**CSS Classes Added:**
```css
.line.original-text {
    background-color: #eff6ff;  /* Light blue */
    border-left: 4px solid #667eea;  /* Purple */
}

.line.modern-text {
    background-color: #f9fafb;  /* Light gray */
    border-left: 4px solid #9ca3af;  /* Gray */
}
```

Both versions include labels: "(Original)" and "(Modern)"

### 5. Implemented View Logic ✅

**JavaScript Changes:**

**New Variables:**
```javascript
let modernPlayData = null;  // Stores modern translation
let viewMode = 'original';  // Current view mode
```

**Data Loading:**
- Loads both JSON files simultaneously using `Promise.all()`
- Loads from local files: `romeo_and_juliet.json` and `romeo_and_juliet_modern.json`

**Display Logic:**
```javascript
function displayScene(act, scene) {
    // Gets corresponding modern scene
    // Displays based on viewMode:
    // - 'original': Original text only
    // - 'modern': Modern text only
    // - 'interleaved': Both, alternating
}
```

**Persistence:**
- Saves view mode preference to localStorage
- Restores preference on page load

## How to Use

### For Users:

1. **Open `indexRegular.html`** in a browser
2. **Select an Act and Scene** from the dropdowns
3. **Click the view mode button** to cycle through:
   - 📜 Original → 📖 Modern → ⇅ Interleaved → (repeat)
4. **Your preference is saved** automatically

### Interleaved View Example:

```
────────────────────────────────────────────
ROMEO (Original)
Two households, both alike in dignity,
In fair Verona, where we lay our scene...
────────────────────────────────────────────
ROMEO (Modern)
In the beautiful city of Verona, where our
story takes place, two families of equal
status...
────────────────────────────────────────────
```

## File Structure

```
grab_shakespeare_reader_v2/
├── romeo_and_juliet.txt              # Original source
├── romeo_and_juliet_modern.txt       # Modern source
├── romeo_and_juliet.json             # Original JSON (NEW)
├── romeo_and_juliet_modern.json      # Modern JSON (NEW)
├── indexRegular.html                 # Updated with interleaved view
├── text_to_json_v2.py                # Conversion script
└── INTERLEAVED_VIEW_SETUP.md         # This file
```

## Features

✅ **Three view modes** (Original, Modern, Interleaved)
✅ **Visual distinction** (colored backgrounds, borders)
✅ **Clear labels** showing which version is displayed
✅ **Persistent preferences** (saved to localStorage)
✅ **Line-by-line matching** between original and modern
✅ **Dark mode support** for all view modes
✅ **All existing features work** (highlighting, search, etc.)

## Technical Details

### Data Structure Matching

Both JSON files have identical structure:
```json
{
  "title": "Romeo and Juliet",
  "author": "William Shakespeare",
  "acts": [
    {
      "act_number": 1,
      "scenes": [
        {
          "scene_number": 1,
          "lines": [
            {
              "speaker": "ROMEO",
              "text": "..."
            }
          ]
        }
      ]
    }
  ]
}
```

The interleaved view matches by:
- Act index
- Scene index
- Line index

### Browser Compatibility

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Uses ES6 features (arrow functions, async/await)
- localStorage for persistence
- CSS Grid/Flexbox for layout

## Next Steps (Optional)

### If you want to add more plays:

1. **Convert text to JSON:**
   ```bash
   python text_to_json_v2.py hamlet.txt hamlet.json "Hamlet"
   python text_to_json_v2.py hamlet_modern.txt hamlet_modern.json "Hamlet (Modern)"
   ```

2. **Add play selector back:**
   - Modify `loadPlayData()` to accept play name parameter
   - Create dropdown with play options
   - Load corresponding original + modern JSONs

3. **Dynamic loading:**
   - Load plays on demand instead of all at once
   - Reduce initial page load time

## Testing

**To test the implementation:**

1. Open `indexRegular.html` in a browser
2. Select "Act 1" from the Act dropdown
3. Select "Scene 1" from the Scene dropdown
4. Click the view mode button multiple times
5. Verify:
   - Original shows only original text
   - Modern shows only modern text
   - Interleaved shows both with proper styling
   - Preference persists after page reload

## Troubleshooting

**If the page doesn't load:**
- Check browser console for errors
- Verify both JSON files exist in the same directory
- Check file paths in `loadPlayData()` function

**If one version is missing:**
- Verify both JSON files were created successfully
- Check JSON structure matches expected format
- Verify line counts match between original and modern

**If styling looks wrong:**
- Check CSS classes are applied correctly
- Verify dark mode toggle works
- Check browser developer tools for CSS conflicts

## Credits

- Original Shakespeare text from various sources
- Modern translation manually created
- Interleaved view inspired by Bible app implementation
- Built with vanilla JavaScript, no frameworks required
