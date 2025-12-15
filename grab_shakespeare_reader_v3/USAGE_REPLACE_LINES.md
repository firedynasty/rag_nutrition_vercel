# Replace Lines Script - Usage Guide

This script helps you manually build your modern Shakespeare translation by replacing specific line ranges with clipboard content.

## Installation

First, install the required dependency:

```bash
pip install pyperclip
```

## Workflow for Building Modern Translation

### Step 1: Identify Lines to Translate

1. Open `romeo_and_juliet.txt` (original text)
2. Find a section you want to translate (e.g., lines 421-444)
3. Note the line numbers

### Step 2: Get Modern Translation

**Option A: Ask Claude for glossing**
```
Copy lines from romeo_and_juliet.txt and ask Claude:
"Please provide a modern translation of this Shakespeare passage"
```

**Option B: Use No Fear Shakespeare**
- Copy the modern translation from No Fear Shakespeare website
- Or use existing modern text you have

### Step 3: Copy Modern Text to Clipboard

- Select and copy the modern translation text
- Make sure it's in your clipboard (Cmd+C on Mac)

### Step 4: Run the Script

```bash
# Basic usage (uses default file: romeo_and_juliet_no_fear.txt)
python replace_lines.py 421 444

# Specify a different file
python replace_lines.py 421 444 my_custom_file.txt
```

### Step 5: Review and Confirm

The script will show you:
- What content will be replaced (OLD CONTENT)
- What the new content will be (NEW CONTENT)
- Number of lines being replaced

Type `y` to confirm, `n` to cancel.

## Example Session

```bash
$ python replace_lines.py 421 444

📄 Using default file: romeo_and_juliet_no_fear.txt

📋 Clipboard contains 523 characters
   (12 lines)

✅ Backup created: romeo_and_juliet_no_fear.txt.bak

📋 Replacing lines 421-444 (24 lines)
   with 12 lines from clipboard

🗑️  OLD CONTENT:
────────────────────────────────────────────────────────────
   421 │ JULIET
   422 │
   423 │ O Romeo, Romeo! wherefore art thou Romeo?
   ...
────────────────────────────────────────────────────────────

✨ NEW CONTENT:
────────────────────────────────────────────────────────────
   421 │ JULIET
   422 │
   423 │ Oh Romeo, Romeo, why do you have to be Romeo?
   ...
────────────────────────────────────────────────────────────

❓ Proceed with replacement? (y/n): y

✅ Successfully replaced lines 421-444 in 'romeo_and_juliet_no_fear.txt'
   Original file backed up to 'romeo_and_juliet_no_fear.txt.bak'
```

## Safety Features

1. **Automatic Backup**: Creates `.bak` file before any changes
2. **Preview**: Shows old and new content before replacing
3. **Confirmation**: Asks for confirmation before making changes
4. **Validation**: Checks line numbers are valid

## Tips

### Finding Line Numbers

Use a text editor that shows line numbers:
- VSCode: Line numbers shown on left
- Sublime Text: View → Line Numbers
- Vim: `:set number`

### Batch Processing

You can process the file section by section:

```bash
# Translate Act 1, Scene 1
python replace_lines.py 32 150

# Translate Act 1, Scene 2
python replace_lines.py 151 250

# And so on...
```

### Restoring from Backup

If something goes wrong:

```bash
# Restore the original
cp romeo_and_juliet_no_fear.txt.bak romeo_and_juliet_no_fear.txt
```

## Common Workflow with Claude

1. **Select original text** (lines 421-444 from romeo_and_juliet.txt)
2. **Ask Claude**:
   ```
   Please gloss this Shakespeare passage with brief definitions
   in parentheses after archaic words, and provide a modern
   translation:
   
   [paste original text]
   ```
3. **Copy Claude's modern translation** to clipboard
4. **Run script**: `python replace_lines.py 421 444`
5. **Confirm** and move to next section

## File Structure

After building your modern translation, you'll have:
- `romeo_and_juliet.txt` - Original text (unchanged)
- `romeo_and_juliet_no_fear.txt` - Modern translation (being built)
- `romeo_and_juliet_no_fear.txt.bak` - Backup before last change

## Next Step: Build JSON

Once you've completed the modern translation file, you'll need a script to:
1. Parse both `romeo_and_juliet.txt` and `romeo_and_juliet_no_fear.txt`
2. Match them line-by-line
3. Generate `romeo_juliet_with_modern.json` with both versions

Let me know when you're ready for that script!



# Automatic Line Balancing Examples

The updated `replace_lines.py` script now automatically balances lines to match your target count!

## How It Works

### Algorithm Priority:

1. **Sentence boundaries** (. ! ?) - Splits at natural sentence breaks first
2. **Commas** - If more splits needed, breaks at commas
3. **Word count** - For long phrases, splits at midpoint
4. **Smart merging** - Combines shortest adjacent lines when reducing

## Your Example

**Your clipboard text (12 lines):**

```
Yes, this is what love does. My sadness sits
heavy in my chest, and you want to add your
own sadness to mine so there's even more.
have too much sadness already, and now you're
going to make me sadder by feeling sorry for
you. Here's what love is: a smoke made out of
lovers' sighs. When the smoke clears, love is a
fire burning in your lover's eyes. If you frustrate
love, you get an ocean made out of lovers' tears.
What else is love? It's a wise form of madness.
It's a sweet lozenge that you choke on.
Goodbye, cousin
```

**Command:**

```bash
python replace_lines.py 100 109
```

(This targets 10 lines: 100-109 inclusive)

**What happens:**

```
⚠️  Clipboard has 12 lines, but target is 10
   🔀 Merging lines to fit...

✨ NEW CONTENT (10 lines):
────────────────────────────────────────────────────────────
  100 │ Yes, this is what love does. My sadness sits heavy in my chest, and you want to add your
  101 │ own sadness to mine so there's even more. have too much sadness already, and now you're
  102 │ going to make me sadder by feeling sorry for you.
  103 │ Here's what love is: a smoke made out of lovers' sighs.
  104 │ When the smoke clears, love is a fire burning in your lover's eyes.
  105 │ If you frustrate love, you get an ocean made out of lovers' tears.
  106 │ What else is love?
  107 │ It's a wise form of madness.
  108 │ It's a sweet lozenge that you choke on.
  109 │ Goodbye, cousin
────────────────────────────────────────────────────────────
```

The script **automatically merged** the 12 lines down to 10 by combining the shortest adjacent lines!

## More Examples

### Example 1: Too Few Lines (Add Fillers)

**Clipboard (3 lines):**

```
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer.
The slings and arrows of outrageous fortune.
```

**Target: 8 lines**

**Result:** Script automatically adds filler lines to reach 8 lines:

```
  1 │ To be or not to be, that is the question.
  2 │ Whether 'tis nobler in the mind to suffer.
  3 │ The slings and arrows of outrageous fortune.
  4 │ **************************************************
  5 │ **************************************************
  6 │ **************************************************
  7 │ **************************************************
  8 │ **************************************************
```

### Example 2: Too Many Lines (Merge)

**Clipboard (15 lines):**

```
Love
is
a
smoke
made
with
the
fume
of
sighs
...
```

**Target: 5 lines**

**Result:** Script merges shortest adjacent lines:

```
  1 │ Love is a
  2 │ smoke made with
  3 │ the fume of
  4 │ sighs
  5 │ ...
```

## Usage Tips

### When Auto-Balance Works Best:

✅ **Good scenarios:**

- Small differences (±3 lines)
- Natural prose with punctuation
- Text with clear sentence/phrase boundaries

⚠️ **Less ideal scenarios:**

- Very large differences (12 lines → 2 lines)
- Poetry with specific line breaks
- Technical text without punctuation

### Manual Control:

If auto-balance doesn't work well, you can:

1. **Manually adjust clipboard** before running script
2. **Edit the result** after it's inserted
3. **Use different line ranges** to better match your content

## Algorithm Details

### Padding Strategy (Too Few Lines):

1. Keep all your original lines intact
2. Add filler lines: `**************************************************`
3. You can easily spot and replace them later
4. Simple and non-destructive

### Merging Strategy (Too Many Lines):

1. Find two **shortest adjacent lines**
2. Combine them with a space
3. Repeat until target count reached
4. Preserves most of the original line structure

## Preview Before Confirm

The script **always shows you** the result before applying:

- Shows OLD content (what will be replaced)
- Shows NEW content (balanced result)
- Asks for confirmation (y/n)

You can cancel (type `n`) if the auto-balance doesn't look good!

## Real Workflow Example

```bash
# 1. You want to replace lines 421-430 (10 lines)
# 2. Claude gives you modern text, but it's 12 lines
# 3. Copy Claude's response
# 4. Run script:
python replace_lines.py 421 430

# Output:
📋 Clipboard contains 523 characters
   (12 lines)

⚠️  Clipboard has 12 lines, but target is 10
   🔀 Merging lines to fit...

📋 Replacing lines 421-430 (10 lines)
   with 10 lines from clipboard
   ✨ Lines were merged to match target count

[Shows preview of OLD and NEW content]

❓ Proceed with replacement? (y/n): y

✅ Successfully replaced lines 421-430
```

No need to manually count or adjust - the script handles it automatically!
