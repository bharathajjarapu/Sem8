# Notes A4 PDF Export

Use this workflow to produce stable, readable A4 PDFs from Markdown notes.

## Verify Tools
Run:

```powershell
pandoc --version
Test-Path "C:\Users\Administrator\AppData\Local\imput\Helium\Application\chrome.exe"
```

If either check fails, stop and fix tool availability first.

## Export Workflow
1. Create a print CSS tuned for A4.
2. Convert Markdown to standalone HTML using Pandoc with math-safe parsing.
3. Print HTML to PDF via Helium Chrome headless.
4. Validate output size and keep desired files.

```powershell
pandoc -f markdown+tex_math_single_backslash -t html5 --mathml input.md `
  -o output.print.html `
  --standalone `
  --css output.export.css `
  --metadata title="Notes"

& "C:\Users\Administrator\AppData\Local\imput\Helium\Application\chrome.exe" `
  --headless=new `
  --disable-gpu `
  --disable-extensions `
  --allow-file-access-from-files `
  --virtual-time-budget=20000 `
  --no-pdf-header-footer `
  --print-to-pdf="C:\path\to\output.pdf" `
  "file:///C:/path/to/output.print.html"
```

## A4 CSS Rules
Use these defaults:
- `@page` size A4 with margins near `12mm 11mm`
- Font around `10.5pt`, line-height around `1.42`
- Avoid large vertical margins around paragraphs/images/tables
- Keep table font slightly smaller than body text
- Do not hard-force page breaks around large images unless absolutely needed

Example large-image control:

```css
img[src*="image_8.png"],
img[src*="image_9.png"],
img[src*="image_10.png"],
img[src*="image_12.png"] {
  width: 96%;
  max-height: 200mm;
  object-fit: contain;
}
```

## Quality Checks
Confirm images are linked in HTML:

```powershell
Select-String -Path output.print.html -Pattern 'images/image_\d+\.png'
```

Optional PDF sanity checks:

```powershell
(Get-Item output.pdf).Length
```

If PDF is unexpectedly tiny for image-heavy notes, regenerate and ensure:
- `output.print.html` exists before browser export
- local image paths are valid
- `--virtual-time-budget` is set (for slower rendering)
- For `\[...\]` or `\(...\)` formulas, ensure Pandoc flags are:
  `-f markdown+tex_math_single_backslash -t html5 --mathml`

## Cleanup Options
Keep only final PDF:

```powershell
Remove-Item output.print.html, output.export.css -ErrorAction SilentlyContinue
```

Or keep PDF + HTML + CSS for future quick re-exports.
