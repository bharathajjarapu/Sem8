# MDtoPDF (A4 Notes Export)

## Purpose
Use this when converting Markdown notes (with tables, formulas, and local images) into clean A4 PDFs.

## Reliable Workflow
1. Verify tools:
   - `pandoc --version`
   - Check browser path:
     `C:\Users\Administrator\AppData\Local\imput\Helium\Application\chrome.exe`
2. Convert Markdown -> HTML with Pandoc using math-safe parsing.
3. Convert HTML -> PDF with Helium Chrome headless.
4. Use `--no-pdf-header-footer` to remove filename/date print headers.
5. Use `--virtual-time-budget` so local images and CSS fully load before print.
6. Validate output size/pages if needed.
7. Keep only final artifacts you want (PDF only, or PDF + export HTML/CSS).

## Standard Commands
```powershell
# 1) Markdown -> print HTML
pandoc -f markdown+tex_math_single_backslash -t html5 --mathml input.md `
  -o output.print.html `
  --standalone `
  --css print.css `
  --metadata title="Document Title"

# 2) HTML -> PDF
& "C:\Users\Administrator\AppData\Local\imput\Helium\Application\chrome.exe" `
  --headless=new `
  --disable-gpu `
  --disable-extensions `
  --allow-file-access-from-files `
  --virtual-time-budget=20000 `
  --no-pdf-header-footer `
  --print-to-pdf="C:\path\to\output.pdf" `
  "file:///C:/path/to/output.print.html"

# 3) Optional checks
(Get-Item "output.pdf").Length
```

## A4 CSS Baseline
```css
@page { size: A4; margin: 12mm 11mm; }
html, body {
  background: #fff;
  color: #111827;
  font-family: "Segoe UI", Calibri, Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.42;
}
body { max-width: 186mm; margin: 0 auto; }
h1, h2, h3 { page-break-after: avoid; break-after: avoid-page; margin: .65em 0 .28em; color: #0f172a; }
h1 { font-size: 18pt; border-bottom: 1.5px solid #dbe3ee; padding-bottom: 6px; }
h2 { font-size: 13pt; }
h3 { font-size: 11.2pt; }
p { margin: .22em 0 .45em; }
ul, ol { margin: .18em 0 .45em 1.2em; }
table { width: 100%; border-collapse: collapse; margin: 6px 0 9px; page-break-inside: avoid; break-inside: avoid-page; font-size: 9.6pt; }
th, td { border: 1px solid #cfd8e3; padding: 5px 6px; vertical-align: top; }
th { background: #f3f6fb; font-weight: 600; }
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 5px auto 8px;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  page-break-inside: avoid;
  break-inside: avoid-page;
}
```

## For Long Flowcharts
Do not force page-break before and after every large image (it wastes space). Prefer size control:
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

## Troubleshooting
- If PDF is suspiciously tiny (for example ~20-30 KB), export likely failed or content was not loaded.
- Ensure `.print.html` exists before running browser print.
- Use `--virtual-time-budget=20000` for image-heavy notes.
- If formulas look broken (for example `\frac` disappears or `\[ ... \]` prints as plain text), ensure Pandoc uses:
  `-f markdown+tex_math_single_backslash -t html5 --mathml`
- Confirm image links in HTML:
  `Select-String -Path output.print.html -Pattern 'images/image_\\d+\\.png'`

## Cleanup
```powershell
Remove-Item output.print.html, print.css -ErrorAction SilentlyContinue
```
