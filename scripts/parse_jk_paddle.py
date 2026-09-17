"""
PaddleOCR-based parser for Jammu & Kashmir (ECI state code U08) unsearchable/
rasterized electoral roll PDFs, following the same box-grid template as
parse_jk.py (3 elector boxes per row, 10 rows per page).

Reuses the tesseract version's cv2 box-grid detection (OCR-agnostic) but
replaces every OCR call with PaddleOCR, which does real per-region text
detection instead of assuming one linear reading block. This fixes two
failure modes observed with tesseract --psm 6:
  - table cells on the same printed line (e.g. "Assembly Constituency ..."
    and "Part No.: N" side by side) sometimes get silently dropped instead
    of merely being read out of order;
  - small/noisy crops (serial number box, gender word) are more often
    misread or left blank.

Usage:
    python3 parse_jk_paddle.py <input.pdf> <output.csv>
    python3 parse_jk_paddle.py --input-dir <pdf folder> --output-dir <csv folder>
"""
import re

import numpy as np
import pandas as pd
import pdf2image
from paddleocr import PaddleOCR

from parse_jk import get_boxes, clean, strip_label, COLUMNS, DPI

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                          use_textline_orientation=False, lang='en')
    return _OCR


def ocr_lines(im):
    """Run PaddleOCR on im; return detected text regions sorted top-to-bottom,
    left-to-right, each as {'top','bottom','left','right','text'}."""
    result = get_ocr().predict(np.array(im))
    items = []
    for res in result:
        for text, poly in zip(res['rec_texts'], res['rec_polys']):
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            items.append({'top': min(ys), 'bottom': max(ys),
                           'left': min(xs), 'right': max(xs), 'text': text})
    items.sort(key=lambda d: (d['top'], d['left']))
    return items


def cluster_rows(items, tol=20):
    """Group text regions into visual rows, tolerating a few pixels of
    vertical jitter between a label and its value on the same printed
    line (a plain top-sort can flip their order and break regexes that
    expect "label then value")."""
    rows = []
    for it in sorted(items, key=lambda d: d['top']):
        for row in rows:
            if abs(row[0]['top'] - it['top']) < tol:
                row.append(it)
                break
        else:
            rows.append([it])
    for row in rows:
        row.sort(key=lambda d: d['left'])
    rows.sort(key=lambda row: sum(it['top'] for it in row) / len(row))
    return rows


def full_text_from_items(items, tol=20):
    return '\n'.join(' '.join(it['text'] for it in row) for row in cluster_rows(items, tol=tol))


def find_line(items, substr, exact=False):
    for it in items:
        if exact:
            if it['text'].strip() == substr:
                return it
        elif substr.lower() in it['text'].lower():
            return it
    return None


def band_text(items, top, bottom, left_min=None, left_max=None):
    sub = [it for it in items if top <= it['top'] <= bottom]
    if left_min is not None:
        sub = [it for it in sub if it['left'] >= left_min]
    if left_max is not None:
        sub = [it for it in sub if it['left'] <= left_max]
    sub.sort(key=lambda d: d['left'])
    return clean(' '.join(it['text'] for it in sub))


def label_value(items, anchor, tol=15, left_min=2300, exact=False):
    row = find_line(items, anchor, exact=exact)
    if row is None:
        return ''
    # Label and value are sometimes merged into a single detection.
    m = re.search(re.escape(anchor) + r'\s*[:\-]\s*(.+)', row['text'], re.IGNORECASE)
    if m and clean(m.group(1)):
        return clean(m.group(1))
    val = band_text(items, row['top'] - tol, row['top'] + tol, left_min=left_min)
    return re.sub(r'^[:\-\s]+', '', val).strip()


def nearest_value_below(items, ref_left, ref_right, min_top, max_top):
    ref_center = (ref_left + ref_right) / 2
    candidates = [it for it in items
                  if min_top <= it['top'] <= max_top and re.fullmatch(r'\d+', it['text'].strip())]
    if not candidates:
        return ''
    best = min(candidates, key=lambda it: abs((it['left'] + it['right']) / 2 - ref_center))
    return best['text'].strip()


def extract_counts(items):
    anchor = find_line(items, 'NUMBER OF ELECTORS') or find_line(items, 'Serial No.')
    top0 = anchor['top'] if anchor else 0

    starting_hdr = find_line(items, 'Starting')
    ending_hdr = find_line(items, 'Ending')
    male_hdr = find_line(items, 'Male', exact=True)
    female_hdr = find_line(items, 'Female', exact=True)
    third_hdr = find_line(items, 'Third Gender')
    total_hdr = find_line(items, 'Total', exact=True)

    def val_for(hdr):
        if hdr is None:
            return ''
        return nearest_value_below(items, hdr['left'], hdr['right'], top0, top0 + 700)

    return {
        'starting_serial_no': val_for(starting_hdr),
        'ending_serial_no': val_for(ending_hdr),
        'net_electors_male': val_for(male_hdr),
        'net_electors_female': val_for(female_hdr),
        'net_electors_third_gender': val_for(third_hdr),
        'net_electors_total': val_for(total_hdr),
    }


def parse_first_page(page):
    items = ocr_lines(page)
    full = full_text_from_items(items)

    ac = re.search(r'Assembly Constituency\s*:\s*(\d+)\s*-\s*(.*?)\s*\(([^)]*)\)', full)
    ac_no, ac_name = (ac.group(1), clean(ac.group(2))) if ac else ('', '')

    pc = re.search(r'Parliamentary Constituency\s*:\s*(\d+)\s*-\s*(.*?)\s*\(([^)]*)\)', full)
    pc_no, pc_name = (pc.group(1), clean(pc.group(2))) if pc else ('', '')

    # Independent of the AC regex above: a table-cell OCR miss on one no
    # longer blanks out the other (see module docstring).
    part_no = ''.join(re.findall(r'Part No\.?\s*:\s*(\d+)', full))
    year = ''.join(re.findall(r'Year of Revision\s*(\d{4})', full))

    fields = {}
    for key, anchor in [
        ('main_town', 'Village'),
        ('ward', 'Ward'),
        ('post_office', 'Office'),
        ('police_station', 'Police'),
        ('panchayat', 'Panchayat'),
        ('municipal_body', 'Block'),
        ('mandal', 'Tehsil/Mandal'),
        ('district', 'District'),
        ('pin_code', 'code'),
    ]:
        fields[key] = label_value(items, anchor)

    ps_no_name = band_text(items, 2950, 3140, left_min=0, left_max=915)
    ps = re.match(r'(\d+)\s*-\s*(.*)', ps_no_name)
    ps_no, ps_name = (ps.group(1), clean(ps.group(2))) if ps else ('', ps_no_name)

    ps_addr = band_text(items, 3200, 3400, left_min=0, left_max=915)

    ps_type_row = find_line(items, 'General', exact=True)
    ps_type = ps_type_row['text'] if ps_type_row else ''

    return {
        'ac_no': ac_no, 'ac_name': ac_name,
        'parl_constituency_no': pc_no, 'parl_constituency_name': pc_name,
        'part_no': part_no, 'year': year,
        **fields,
        'polling_station_no': ps_no, 'polling_station_name': ps_name,
        'polling_station_address': ps_addr, 'polling_station_type': ps_type,
        **extract_counts(items),
    }


def parse_page_header(page):
    im = np.array(page)
    header_crop = im[0:220, :]
    items = ocr_lines(header_crop)
    full = full_text_from_items(items)

    sec = re.search(r'Section No and Name\s*(\d+)\s*-\s*(.*)', full)
    section_no, section_name = (sec.group(1), clean(sec.group(2))) if sec else ('', '')

    return {'section_no': section_no, 'section_name': section_name}


REL_PREFIXES = [('husband', 'husband'), ('father', 'father'), ('mother', 'mother')]


def parse_box(box):
    items = ocr_lines(box)

    number = ''
    voter_id = ''
    name = ''
    rel_type = ''
    rel_name = ''
    house_no = ''
    age = ''
    sex = ''

    for it in items:
        s = clean(it['text'])
        if not s:
            continue
        low = s.lower()

        if number == '' and re.fullmatch(r'\d{1,4}', s):
            number = s
            continue

        if voter_id == '' and re.fullmatch(r'[A-Z]{2,4}\d{6,10}', s.replace(' ', '').upper()):
            voter_id = s.replace(' ', '').upper()
            continue

        if low.startswith('name'):
            name = strip_label(s)
            continue

        matched_rel = False
        for prefix, rtype in REL_PREFIXES:
            if low.startswith(prefix):
                rel_type = rtype
                rel_name = strip_label(s)
                matched_rel = True
                break
        if matched_rel:
            continue

        if 'house' in low and 'number' in low:
            nums = re.findall(r'\d+', s)
            house_no = nums[0] if nums else house_no
            continue

        if low.startswith('age') or 'gender' in low:
            nums = re.findall(r'\d+', s)
            if nums:
                age = nums[0]
            gm = re.search(r'gender\s*[:\-]?\s*([A-Za-z]+)', s, re.IGNORECASE)
            if 'third' in low:
                sex = 'Third Gender'
            elif gm:
                g = gm.group(1).lower()
                if g.startswith('f'):
                    sex = 'Female'
                elif g.startswith('m'):
                    sex = 'Male'
            continue

    return {
        'number': number,
        'id': voter_id,
        'elector_name': name,
        'father_or_husband_name': rel_name,
        'relation_type': rel_type,
        'house_no': house_no,
        'age': age,
        'sex': sex,
    }


def process_pdf(pdf_path, out_csv, state='jammu_kashmir'):
    info = pdf2image.pdfinfo_from_path(pdf_path)
    total_pages = info['Pages']
    filename = pdf_path.split('/')[-1]

    print(f'Total pages: {total_pages}')

    first_page = pdf2image.convert_from_path(pdf_path, dpi=DPI, first_page=1, last_page=1, fmt='jpg')[0]
    header = parse_first_page(first_page)
    print('Parsed first-page header:', {k: v for k, v in header.items() if k in
          ('ac_name', 'part_no', 'district', 'net_electors_total')})

    grid_first, grid_last = 3, total_pages - 1  # last page is the deletion/remarks summary

    records = []
    for pg in range(grid_first, grid_last + 1):
        page = pdf2image.convert_from_path(pdf_path, dpi=DPI, first_page=pg, last_page=pg, fmt='jpg')[0]
        page_header = parse_page_header(page)
        boxes = get_boxes(page)
        print(f'Page {pg}: {len(boxes)} boxes, section={page_header.get("section_name")}')

        results = [parse_box(b) for b in boxes]

        for r in results:
            if not any([r['elector_name'], r['id'], r['number']]):
                continue
            rec = {**header, **page_header, **r, 'state': state, 'filename': filename}
            records.append(rec)

    # Serial numbers are small, easily-misread OCR crops (both tesseract and
    # paddle confuse digits like 5/9, 1/4 here), but the roll lists electors
    # in strict gapless order, so position + starting_serial_no is exact
    # where the OCR read is merely noisy.
    start = header.get('starting_serial_no', '')
    base = int(start) if str(start).isdigit() else 1
    for i, rec in enumerate(records):
        rec['number'] = str(base + i)

    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    df.to_csv(out_csv, index=False)
    print(f'Wrote {len(df)} elector records to {out_csv}')
    return df


if __name__ == '__main__':
    from cli import main
    main(process_pdf, 'Parse J&K (U08) unsearchable electoral roll PDF(s) using PaddleOCR')
