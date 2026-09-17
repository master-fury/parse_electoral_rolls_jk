"""
Parser for Jammu & Kashmir (ECI state code U08) unsearchable/rasterized
electoral roll PDFs, following the box-grid template used for the
"FinalRoll ... ENG" exports (3 elector boxes per row, 10 rows per page).

Usage:
    python3 parse_jk.py <input.pdf> <output.csv>
    python3 parse_jk.py --input-dir <pdf folder> --output-dir <csv folder>
"""
import re

import cv2
import numpy as np
import pandas as pd
import pdf2image
import pytesseract

DPI = 200

# Calibrated at DPI=200 on this template
BOX_LIMITS_H = (400, 460)
BOX_LIMITS_W = (1000, 1150)
ROW_TOLERANCE = 60

COLUMNS = [
    'number', 'id', 'elector_name', 'father_or_husband_name', 'relation_type',
    'house_no', 'age', 'sex',
    'ac_no', 'ac_name', 'parl_constituency_no', 'parl_constituency_name',
    'part_no', 'section_no', 'section_name',
    'year', 'state', 'filename',
    'main_town', 'ward', 'post_office', 'police_station', 'panchayat',
    'municipal_body', 'mandal', 'district', 'pin_code',
    'polling_station_no', 'polling_station_name', 'polling_station_address',
    'polling_station_type',
    'starting_serial_no', 'ending_serial_no',
    'net_electors_male', 'net_electors_female', 'net_electors_third_gender',
    'net_electors_total',
]


def ocr(im, psm=6, whitelist=None):
    config = f'--psm {psm}'
    if whitelist:
        config += f' -c tessedit_char_whitelist={whitelist}'
    return pytesseract.image_to_string(im, config=config, lang='eng')


def clean(v):
    return re.sub(r'\s+', ' ', v or '').strip()


def word_boxes(im):
    df = pytesseract.image_to_data(im, config='--psm 6', lang='eng',
                                    output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=['text'])
    return df[df.text.str.strip() != '']


def band(df, top, bottom, left_min=None, left_max=None):
    m = (df.top >= top) & (df.top <= bottom)
    if left_min is not None:
        m = m & (df.left >= left_min)
    if left_max is not None:
        m = m & (df.left <= left_max)
    sub = df[m].sort_values('left')
    return clean(' '.join(sub.text.tolist()))


def find_word(df, substr, exact=False):
    if exact:
        m = df[df.text == substr]
    else:
        m = df[df.text.str.lower().str.contains(substr.lower(), regex=False)]
    return m.iloc[0] if len(m) else None


def label_value(df, anchor, tol=10, left_min=2385, exact=False):
    row = find_word(df, anchor, exact=exact)
    if row is None:
        return ''
    return band(df, row.top - tol, row.top + tol, left_min=left_min)


def parse_first_page(page):
    im = np.array(page)
    h, w = im.shape[:2]

    full_text = ocr(im, psm=6)
    df = word_boxes(im)

    ac = re.search(r'Assembly Constituency\s*:\s*(\d+)\s*-\s*(.*?)\s*\(([^)]*)\)\s*Part', full_text)
    ac_no, ac_name = (ac.group(1), clean(ac.group(2))) if ac else ('', '')

    pc = re.search(r'Parliamentary Constituency\s*:\s*(\d+)\s*-\s*(.*?)\s*\(([^)]*)\)', full_text)
    pc_no, pc_name = (pc.group(1), clean(pc.group(2))) if pc else ('', '')

    part_no = ''.join(re.findall(r'Part No\.?\s*:\s*(\d+)', full_text))

    year = ''.join(re.findall(r'Year of Revision\s*(\d{4})', full_text))

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
        fields[key] = label_value(df, anchor)

    ps_no_name = band(df, 2950, 3140, left_min=0, left_max=915)
    ps = re.match(r'(\d+)\s*-\s*(.*)', ps_no_name)
    ps_no, ps_name = (ps.group(1), clean(ps.group(2))) if ps else ('', ps_no_name)

    ps_addr = band(df, 3200, 3400, left_min=0, left_max=915)

    ps_type_row = find_word(df, 'General', exact=True)
    ps_type = ps_type_row.text if ps_type_row is not None else ''

    # Elector counts data row sits around 72-80% of page height
    table_crop = im[int(h * 0.72):int(h * 0.80), 0:w]
    table_text = ocr(table_crop, psm=6)
    nums = re.findall(r'\d+', table_text)
    start_sn, end_sn, male, female, third, total = (nums + [''] * 6)[:6]

    return {
        'ac_no': ac_no, 'ac_name': ac_name,
        'parl_constituency_no': pc_no, 'parl_constituency_name': pc_name,
        'part_no': part_no, 'year': year,
        **fields,
        'polling_station_no': ps_no, 'polling_station_name': ps_name,
        'polling_station_address': ps_addr, 'polling_station_type': ps_type,
        'starting_serial_no': start_sn, 'ending_serial_no': end_sn,
        'net_electors_male': male, 'net_electors_female': female,
        'net_electors_third_gender': third, 'net_electors_total': total,
    }


def parse_page_header(page):
    im = np.array(page)
    header = im[0:180, :]
    text = ocr(header, psm=6)

    sec = re.search(r'Section No and Name\s*(\d+)\s*-\s*(.*)', text)
    section_no, section_name = (sec.group(1), clean(sec.group(2))) if sec else ('', '')

    return {'section_no': section_no, 'section_name': section_name}


def get_countours(im):
    _, thresh = cv2.threshold(im, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_boxes(page):
    im = np.array(page)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    contours = get_countours(gray)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if BOX_LIMITS_H[0] < h < BOX_LIMITS_H[1] and BOX_LIMITS_W[0] < w < BOX_LIMITS_W[1]:
            dup = any(abs(x - bx) < 50 and abs(y - by) < 50 for bx, by, _, _ in boxes)
            if not dup:
                boxes.append((x, y, w, h))

    # sort row-major: group by y (row), then order by x (column) within row
    boxes.sort(key=lambda b: b[1])
    rows = []
    for b in boxes:
        placed = False
        for row in rows:
            if abs(row[0][1] - b[1]) < ROW_TOLERANCE:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    ordered = []
    for row in rows:
        row.sort(key=lambda b: b[0])
        ordered.extend(row)

    return [im[y:y + h, x:x + w] for x, y, w, h in ordered]


def strip_label(line):
    # Drop a leading "Label <sep>" prefix where <sep> is a colon-like
    # character that OCR sometimes misreads as >, +, *, =, or -.
    m = re.match(r"^[A-Za-z][A-Za-z'\s]*?[:>+*=~-]\s*(.*)$", line)
    if m:
        return clean(m.group(1))
    # Fallback: OCR dropped the separator entirely (e.g. "Name 7 RAM ...").
    # Strip the label word, then a lone 1-2 char junk token if present.
    parts = line.split(None, 1)
    if len(parts) == 2:
        rest = parts[1]
        head = rest.split(None, 1)[0]
        if len(head) <= 2:
            rest = re.sub(r'^\S{1,2}\s+', '', rest)
        return clean(rest)
    return clean(line)


def parse_box(box):
    h, w = box.shape[:2]

    x, y, bw, bh = 19, 22, 329, 63
    serial_crop = box[y + 3:y + bh - 3, x + 3:x + bw - 3]
    serial_crop = cv2.resize(serial_crop, (serial_crop.shape[1] * 4, serial_crop.shape[0] * 4),
                              interpolation=cv2.INTER_CUBIC)
    serial_gray = cv2.cvtColor(serial_crop, cv2.COLOR_BGR2GRAY)
    _, serial_bw = cv2.threshold(serial_gray, 150, 255, cv2.THRESH_BINARY)
    serial_text = ocr(serial_bw, psm=7, whitelist='0123456789')
    number = ''.join(re.findall(r'\d+', serial_text))

    id_crop = box[0:95, 360:w]
    id_text = ocr(id_crop, psm=11)
    voter_id = ''.join(re.findall(r'[A-Z0-9]{6,}', id_text.replace(' ', '').upper()))

    details_crop = box[95:h, 0:min(805, w)]
    details_text = ocr(details_crop, psm=6)

    name = ''
    rel_name = ''
    rel_type = ''
    house_no = ''
    age = ''
    sex = ''

    for line in details_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        low = line.lower()

        if low.startswith('name'):
            name = strip_label(line)
            continue

        if low.startswith('husband'):
            rel_type = 'husband'
            rel_name = strip_label(line)
            continue

        if low.startswith('father'):
            rel_type = 'father'
            rel_name = strip_label(line)
            continue

        if low.startswith('mother'):
            rel_type = 'mother'
            rel_name = strip_label(line)
            continue

        if 'house' in low and 'number' in low:
            nums = re.findall(r'\d+', line)
            house_no = nums[0] if nums else ''
            continue

        if low.startswith('age') or ('age' in low and 'gender' in low):
            nums = re.findall(r'\d+', line)
            age = nums[0] if nums else ''
            # Gender value is OCR-noisy ("Femaie", "Fermnale", "Mata");
            # the leading letter of the trailing token is far more
            # reliable than a whole-word match.
            tail = re.split(r'[:>+*]', line)[-1].strip().lower()
            if 'third' in tail:
                sex = 'Third Gender'
            elif tail[:1] == 'f':
                sex = 'Female'
            elif tail[:1] == 'm':
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

        # NOTE: threading tesseract subprocess calls here causes severe
        # contention/slowdown in this environment; sequential is far faster.
        results = [parse_box(b) for b in boxes]

        for r in results:
            if not any([r['elector_name'], r['id'], r['number']]):
                continue
            rec = {**header, **page_header, **r, 'state': state, 'filename': filename}
            records.append(rec)

    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    df.to_csv(out_csv, index=False)
    print(f'Wrote {len(df)} elector records to {out_csv}')
    return df


if __name__ == '__main__':
    from cli import main
    main(process_pdf, 'Parse Jammu & Kashmir (U08) unsearchable electoral roll PDF(s) using Tesseract')
