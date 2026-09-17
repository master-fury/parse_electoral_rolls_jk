# parse_electoral_rolls_jk

Parse Jammu & Kashmir electoral rolls (ECI state code **U08**) into CSV for
research and analysis.

The published rolls are scanned/rasterized PDFs with no text layer -- every
elector's details (name, relation, house number, age, sex, voter ID, ...)
sit inside a fixed 3-boxes-per-row grid template, and the only way to get
the data out is OCR. This repo detects that box grid with OpenCV and OCRs
each field, alongside the per-part header (constituency, polling station,
district, elector counts, ...).

## Repo layout

```
scripts/
  parse_jk.py         # Tesseract-based parser
  parse_jk_paddle.py  # PaddleOCR-based parser (more accurate, much slower)
  cli.py              # shared command-line entry point used by both
notebooks/
  quickstart.ipynb    # single-file / batch / inspect-output walkthrough
samples/
  *.pdf                       # one example roll PDF (add your own here for batch testing)
  reference_output/           # example CSVs produced by each parser for that PDF
output/                # your generated CSVs land here (git-ignored)
requirements.txt
```

## Which parser should I use?

| | `parse_jk.py` (Tesseract) | `parse_jk_paddle.py` (PaddleOCR) |
|---|---|---|
| Speed | ~5-10s/page | ~30-60s/page (CPU, no GPU used) |
| Accuracy | good; occasional blank/misread fields | noticeably fewer blanks and misreads |
| Setup | needs the `tesseract` system binary | pure pip install, downloads its own models on first run |

Both expose the identical `process_pdf(pdf_path, out_csv) -> pandas.DataFrame`
function and CSV schema, so you can run both over the same file and diff the
results. If you're only OCR'ing a handful of rolls and want the best data
quality, use PaddleOCR. If you're iterating on the parsing logic itself or
processing many files, start with Tesseract.

[`samples/reference_output/jk_79_8_tesseract.csv`](samples/reference_output/jk_79_8_tesseract.csv)
and [`jk_79_8_paddle.csv`](samples/reference_output/jk_79_8_paddle.csv) are
both parsers' actual output for the sample PDF, so you can diff them
directly to see the accuracy difference for yourself.

## Installation

### 1. Python environment (all platforms)

```bash
python3 -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

```cmd
:: Windows (cmd.exe)
.venv\Scripts\activate.bat
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

This pulls in both the Tesseract path (`pytesseract`, `opencv-python-headless`)
and the PaddleOCR path (`paddlepaddle`, `paddleocr`). Comment out whichever
pair you don't need in `requirements.txt` if you want a lighter install.

### 2. System dependencies

`pdf2image` needs **poppler**, and `parse_jk.py` needs the **tesseract**
binary. `parse_jk_paddle.py` needs neither -- PaddleOCR ships as pure Python
wheels and downloads its own OCR models on first run.

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-eng
```

**macOS (Homebrew):**

```bash
brew install poppler tesseract
```

**Windows:**

- **poppler**: download a prebuilt binary from
  [oschwartz10612/poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases)
  (the build referenced in `pdf2image`'s own docs), unzip it, and add its
  `Library\bin` folder to your `PATH`. (Alternatively, if you use conda:
  `conda install -c conda-forge poppler`.)
- **tesseract**: install from the
  [UB-Mannheim Windows build](https://github.com/UB-Mannheim/tesseract/wiki)
  (the standard Windows distribution referenced in `pytesseract`'s own docs).
  The installer includes English data by default and offers to add itself to
  `PATH` -- accept that, or add the install directory (e.g.
  `C:\Program Files\Tesseract-OCR`) to `PATH` manually.

Verify both are on `PATH` in a fresh terminal:

```bash
pdftoppm -v
tesseract --version
```

## Usage

All commands below assume your virtual environment is activated and your
terminal's working directory is the repo root. On Windows, use `python`
instead of `python3` if that's how Python is set up on your system; every
other flag is identical across Linux/macOS/Windows.

### Single file

```bash
python3 scripts/parse_jk_paddle.py samples/2024-EROLLGEN-U08-79-FinalRoll-Revision2-ENG-8-WI.pdf output/jk_79_8.csv
```

Swap in `scripts/parse_jk.py` for the faster Tesseract-based parser with the
same arguments.

### Multiple files (batch)

Point `--input-dir` at a folder of PDFs and `--output-dir` at where the CSVs
should go; one CSV is written per input PDF, named after it. Only one example
PDF ships in `samples/` (to keep the repo small) -- drop your own additional
roll PDFs into that folder, or any other folder, to batch-process more:

```bash
python3 scripts/parse_jk_paddle.py --input-dir samples --output-dir output
```

```
[1/3] 2024-EROLLGEN-U08-79-FinalRoll-Revision2-ENG-8-WI.pdf -> output/2024-EROLLGEN-U08-79-FinalRoll-Revision2-ENG-8-WI.csv
[2/3] some-other-part.pdf -> output/some-other-part.csv
[3/3] yet-another-part.pdf -> output/yet-another-part.csv
```

If you want everything combined into one CSV afterwards:

```bash
python3 -c "
import pandas as pd, glob
pd.concat([pd.read_csv(f, dtype=str) for f in glob.glob('output/*.csv')], ignore_index=True) \
  .to_csv('output/combined.csv', index=False)
"
```

### Jupyter notebook

See [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) for a runnable
walkthrough of single-file, batch, and output-inspection usage. The short
version -- both parser modules expose a plain function, so you can call it
directly instead of going through the CLI:

```python
import sys
sys.path.insert(0, 'scripts')  # or the absolute path to this repo's scripts/ folder

from parse_jk_paddle import process_pdf  # or: from parse_jk import process_pdf

df = process_pdf('samples/2024-EROLLGEN-U08-79-FinalRoll-Revision2-ENG-8-WI.pdf', 'output/jk_79_8.csv')
df.head()
```

For multiple files in a notebook, loop over `pathlib.Path('samples').glob('*.pdf')`
and call `process_pdf` per file -- see the notebook for the full example.

## Output schema

Each row is one elector. Key columns:

| column | meaning |
|---|---|
| `number` | serial number within the part (derived from the elector's position in the roll, not raw OCR -- see below) |
| `id` | voter ID (EPIC number) |
| `elector_name`, `father_or_husband_name`, `relation_type`, `age`, `sex`, `house_no` | per-elector fields |
| `ac_no`, `ac_name`, `parl_constituency_no`, `parl_constituency_name`, `part_no`, `year` | constituency/part header |
| `district`, `main_town`, `ward`, `post_office`, `police_station`, `panchayat`, `municipal_body`, `mandal`, `pin_code` | part location fields |
| `section_no`, `section_name` | sub-section within the part (varies per page) |
| `polling_station_no`, `polling_station_name`, `polling_station_address`, `polling_station_type` | polling station header |
| `starting_serial_no`, `ending_serial_no`, `net_electors_male`, `net_electors_female`, `net_electors_third_gender`, `net_electors_total` | part-level elector counts, from the first page |
| `state`, `filename` | constant per run, for tracing rows back to source |

## Known limitations

- **Serial numbers are position-derived, not OCR'd.** The tiny serial-number
  crop is misread by both engines often enough (single-digit confusions like
  5/9, 1/4) that `parse_jk_paddle.py` instead numbers electors by their
  position in the roll, offset by the header's `starting_serial_no`. This is
  exact as long as a part's serial numbers are gapless (true for every
  sample checked so far, and expected for a "Final Roll" which is
  renumbered after deletions) -- `parse_jk.py` still uses the raw OCR read.
- **Compound words occasionally lose their space** in PaddleOCR output (e.g.
  `KOTE UPPER` -> `KOTEUPPER`). Cosmetic; doesn't affect matching on IDs or
  numeric fields.
- Both parsers assume the specific box-grid template used by these
  "FinalRoll ... ENG" exports (3 elector boxes per row, 10 rows per page,
  first 2 pages are cover/index, last page is a deletions summary). A
  differently-formatted roll PDF will need recalibrating the box/crop
  geometry at the top of each script.
