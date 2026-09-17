"""Shared command-line entry point for parse_jk.py / parse_jk_paddle.py.

Both parsers expose a `process_pdf(pdf_path, out_csv)` function with the
same signature; this just wires that up to a single-file or batch-folder
command line so the two scripts don't duplicate the same argparse code.
"""
import argparse
from pathlib import Path


def main(process_pdf, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('pdf', nargs='?', help='path to a single roll PDF (omit when using --input-dir)')
    parser.add_argument('out', nargs='?', help='output CSV path for that PDF (omit when using --input-dir)')
    parser.add_argument('--input-dir', help='process every .pdf file in this directory instead of a single file')
    parser.add_argument('--output-dir', help='directory to write one CSV per PDF (required with --input-dir)')
    args = parser.parse_args()

    if args.input_dir:
        if not args.output_dir:
            parser.error('--output-dir is required when using --input-dir')
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pdfs = sorted(Path(args.input_dir).glob('*.pdf'))
        if not pdfs:
            parser.error(f'no .pdf files found in {args.input_dir}')

        for i, pdf in enumerate(pdfs, 1):
            out_csv = out_dir / (pdf.stem + '.csv')
            print(f'[{i}/{len(pdfs)}] {pdf.name} -> {out_csv}')
            process_pdf(str(pdf), str(out_csv))
    else:
        if not args.pdf or not args.out:
            parser.error('provide PDF and OUT, or use --input-dir/--output-dir for batch mode')
        process_pdf(args.pdf, args.out)
