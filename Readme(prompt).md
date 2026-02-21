# Prompt Support (FP5)

## Overview

Added support for reading multiple prompts from a file (`prompt.txt`) so the program can generate multiple images in one run, given that there is a input image folder.

Before this change:
- The program only accepted `--prompt`
- One run generated one image

After this change:
- The program supports `--prompt-file`
- One run can generate multiple images (one per prompt)

---

## How It Works

### New Argument

```bash
--prompt-file prompt.txt
```
### Prompt File Format
```
[1]
First prompt text here

[2]
Second prompt text here

[3]
Third prompt text here
```
### What the Program Does

When `--prompt-file` is used:

- The program reads `prompt.txt`
- It extracts each numbered prompt
- It loops through each prompt
- It generates one image per prompt using the selected style
- It saves all images to `output_images/<style_name>/`

## Sample Run

### Example `prompt.txt`

### Command

```bash
python -m src.main --style noir --prompt-file prompt.txt --steps 10 --height 512 --width 512
