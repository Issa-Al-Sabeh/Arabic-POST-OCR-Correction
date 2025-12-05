import os
import shutil
from pathlib import Path
import json

import cv2
import numpy as np
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from pytesseract import Output

# =============================
# PATHS / SETTINGS
# =============================
# Change this if Poppler is installed elsewhere
POPPLER_PATH = r"C:\poppler\Library\bin"

# Change this if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_LANG = "ara"   # or "ara+eng" if you want both

BASE_DIR = Path(__file__).resolve().parent

INPUT_TXT_DIR = BASE_DIR / "input_txt"
DONE_TXT_DIR = BASE_DIR / "input_txt_done"
OUTPUT_PDF_DIR = BASE_DIR / "output_pdf"
TEMP_PDF_DIR = BASE_DIR / "temp_pdf"
OUTPUT_JSON_DIR = BASE_DIR / "output_json"   # <--- OCR JSON output here

FONT_PATH = BASE_DIR / "Amiri-Regular.ttf"
FONT_NAME = "Arabic"

# Create folders if they don't exist
INPUT_TXT_DIR.mkdir(exist_ok=True)
DONE_TXT_DIR.mkdir(exist_ok=True)
OUTPUT_PDF_DIR.mkdir(exist_ok=True)
TEMP_PDF_DIR.mkdir(exist_ok=True)
OUTPUT_JSON_DIR.mkdir(exist_ok=True)


# =============================
# ARABIC TXT -> CLEAN PDF
# =============================

class PDF(FPDF):
    pass


def prepare_arabic(text: str) -> str:
    """Shape and reorder Arabic text line by line."""
    processed_lines = []
    for line in text.splitlines():
        if not line.strip():
            processed_lines.append("")  # keep empty lines
            continue
        reshaped = arabic_reshaper.reshape(line)
        bidi_line = get_display(reshaped)
        processed_lines.append(bidi_line)
    return "\n".join(processed_lines)


def txt_to_pdf(txt_path: Path, pdf_path: Path):
    """Convert a UTF-8 Arabic .txt file into a clean PDF."""
    pdf = PDF()
    pdf.add_page()

    pdf.add_font(FONT_NAME, "", str(FONT_PATH))
    pdf.set_font(FONT_NAME, size=14)
    pdf.set_auto_page_break(auto=True, margin=15)

    # Align to the right for Arabic
    pdf.set_right_margin(15)
    pdf.set_left_margin(15)

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    arabic_text = prepare_arabic(raw_text)

    # multi_cell will wrap properly; align='R' for RTL
    pdf.multi_cell(0, 8, text=arabic_text, align='R')

    pdf.output(str(pdf_path))
    print(f"  -> Clean PDF saved: {pdf_path.name}")


# =============================
# ADD NOISE / SCAN EFFECT
# =============================

def add_noise_and_scan_effect(img: Image.Image) -> Image.Image:
    op = np.array(img)

    # 1. Slight downscale / upscale (mild low-DPI effect)
    h, w = op.shape[:2]
    new_w = max(1, int(w * 0.75))
    new_h = max(1, int(h * 0.75))
    op = cv2.resize(op, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    op = cv2.resize(op, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Medium Gaussian blur (noticeably softer text)
    op = cv2.GaussianBlur(op, (5, 5), 1.0)

    # 3. Medium Gaussian noise
    noise = np.random.normal(0, 20, op.shape).astype(np.int16)
    op = np.clip(op.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 4. Light erosion (slightly broken / thin characters)
    kernel = np.ones((2, 2), np.uint8)
    op = cv2.erode(op, kernel, iterations=2)

    # 5. Small rotation (a bit misaligned)
    angle = np.random.uniform(-1.0, 1.0)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    op = cv2.warpAffine(op, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 6. Mild vignette / illumination variation
    mask = cv2.GaussianBlur(
        np.random.rand(h, w).astype(np.float32),
        (81, 81),
        0
    )
    mask = 0.85 + 0.15 * (mask / mask.max())  # values ~[0.85, 1.0]
    op = (op.astype(np.float32) * mask[..., None]).astype(np.uint8)

    # 7. Slight yellow tint
    tint = np.full_like(op, (12, 10, 0))
    op = cv2.add(op, tint)

    return Image.fromarray(op)


def clean_pdf_to_noisy(clean_pdf_path: Path, noisy_pdf_path: Path):
    """Convert clean PDF to noisy 'scanned' PDF."""
    pages = convert_from_path(
        str(clean_pdf_path),
        dpi=150,
        poppler_path=POPPLER_PATH
    )

    output_images = [add_noise_and_scan_effect(p) for p in pages]

    # Save as multi-page PDF
    output_images[0].save(
        str(noisy_pdf_path),
        save_all=True,
        append_images=output_images[1:]
    )
    print(f"  -> Noisy PDF saved: {noisy_pdf_path.name}")


# =============================
# OCR: NOISY PDF -> JSON
# =============================

def ocr_pdf_to_custom_json(pdf_path: Path) -> dict:
    """Run OCR on a PDF and return a structured JSON-like dict."""
    pages = convert_from_path(
        str(pdf_path),
        dpi=300,
        poppler_path=POPPLER_PATH
    )

    result = {
        "document": pdf_path.name,
        "total_pages": len(pages),
        "pages": []
    }

    for page_num, img in enumerate(pages, start=1):
        print(f"     - OCR page {page_num} ...", end=" ", flush=True)

        data = pytesseract.image_to_data(
            img,
            lang=TESS_LANG,
            output_type=Output.DICT
        )

        words = []
        all_text_parts = []

        n_items = len(data["text"])
        for i in range(n_items):
            text = data["text"][i].strip()
            if not text:
                continue  # skip empty entries

            conf_str = str(data["conf"][i])
            try:
                conf = int(float(conf_str))
            except ValueError:
                conf = -1

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            words.append({
                "text": text,
                "confidence": conf,
                "bbox": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                }
            })

            all_text_parts.append(text)

        page_text = " ".join(all_text_parts)

        page_entry = {
            "page_number": page_num,
            "text": page_text,
            "word_count": len(words),
            "words": words
        }
        result["pages"].append(page_entry)

        print("done")

    return result


def ocr_pdf_to_json_file(pdf_path: Path, json_path: Path):
    """OCR a PDF and write the result to a JSON file."""
    print(f"  -> OCR (JSON) on: {pdf_path.name}")
    ocr_json = ocr_pdf_to_custom_json(pdf_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)

    print(f"  -> OCR JSON saved: {json_path.name}")


# =============================
# MAIN PIPELINE
# =============================

def process_txt_file(txt_file: Path):
    base_name = txt_file.stem

    clean_pdf_path = TEMP_PDF_DIR / f"{base_name}_clean.pdf"
    noisy_pdf_path = OUTPUT_PDF_DIR / f"{base_name}.pdf"
    ocr_json_path = OUTPUT_JSON_DIR / f"{base_name}.json"

    print(f"\nProcessing: {txt_file.name}")

    # 1) txt -> clean PDF
    txt_to_pdf(txt_file, clean_pdf_path)

    # 2) clean PDF -> noisy PDF
    clean_pdf_to_noisy(clean_pdf_path, noisy_pdf_path)

    # 3) noisy PDF -> OCR JSON
    ocr_pdf_to_json_file(noisy_pdf_path, ocr_json_path)

    # 4) Remove temp clean PDF
    try:
        clean_pdf_path.unlink()
    except FileNotFoundError:
        pass

    # 5) Move txt to done folder
    dest_txt = DONE_TXT_DIR / txt_file.name
    shutil.move(str(txt_file), str(dest_txt))
    print(f"  -> Moved txt to: {dest_txt}")


def main():
    txt_files = sorted(INPUT_TXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {INPUT_TXT_DIR.resolve()}")
        return

    print(f"Found {len(txt_files)} .txt files. Starting pipeline...\n")

    for txt_file in txt_files:
        try:
            process_txt_file(txt_file)
        except Exception as e:
            print(f"❌ Error processing {txt_file.name}: {e}")

    print("\n🎉 Done! Noisy PDFs are in:", OUTPUT_PDF_DIR.resolve())
    print("📄 OCR JSON files are in:", OUTPUT_JSON_DIR.resolve())


if __name__ == "__main__":
    main()
