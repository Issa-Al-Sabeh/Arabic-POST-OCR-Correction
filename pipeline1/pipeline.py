import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# =============================
# PATHS / SETTINGS
# =============================
# Change this if Poppler is installed elsewhere
POPPLER_PATH = r"C:\poppler\Library\bin"

# Change this if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BASE_DIR = Path(__file__).resolve().parent

INPUT_TXT_DIR = BASE_DIR / "input_txt"
DONE_TXT_DIR = BASE_DIR / "input_txt_done"
OUTPUT_PDF_DIR = BASE_DIR / "output_pdf"
TEMP_PDF_DIR = BASE_DIR / "temp_pdf"
OUTPUT_TXT_DIR = BASE_DIR / "output_txt"   # <--- OCR output here

FONT_PATH = BASE_DIR / "Amiri-Regular.ttf"
FONT_NAME = "Arabic"

# Create folders if they don't exist
INPUT_TXT_DIR.mkdir(exist_ok=True)
DONE_TXT_DIR.mkdir(exist_ok=True)
OUTPUT_PDF_DIR.mkdir(exist_ok=True)
TEMP_PDF_DIR.mkdir(exist_ok=True)
OUTPUT_TXT_DIR.mkdir(exist_ok=True)


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
# OCR: NOISY PDF -> TXT
# =============================

def ocr_pdf_to_txt(pdf_path: Path, txt_path: Path):
    """Run Tesseract OCR on a PDF and save text to txt_path."""
    print(f"  -> OCR on: {pdf_path.name}")

    pages = convert_from_path(
        str(pdf_path),
        dpi=300,
        poppler_path=POPPLER_PATH
    )

    all_text = []

    for page_num, img in enumerate(pages, start=1):
        print(f"     - Page {page_num} ...", end=" ", flush=True)
        text = pytesseract.image_to_string(img, lang="ara")  # Arabic OCR
        all_text.append(text)
        print("done")

    full_text = "\n\n\f\n\n".join(all_text)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"  -> OCR txt saved: {txt_path.name}")


# =============================
# MAIN PIPELINE
# =============================

def process_txt_file(txt_file: Path):
    base_name = txt_file.stem

    clean_pdf_path = TEMP_PDF_DIR / f"{base_name}_clean.pdf"
    noisy_pdf_path = OUTPUT_PDF_DIR / f"{base_name}.pdf"
    ocr_txt_path = OUTPUT_TXT_DIR / f"{base_name}.txt"

    print(f"\nProcessing: {txt_file.name}")

    # 1) txt -> clean PDF
    txt_to_pdf(txt_file, clean_pdf_path)

    # 2) clean PDF -> noisy PDF
    clean_pdf_to_noisy(clean_pdf_path, noisy_pdf_path)

    # 3) noisy PDF -> OCR txt
    ocr_pdf_to_txt(noisy_pdf_path, ocr_txt_path)

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
    print("📄 OCR txt files are in:", OUTPUT_TXT_DIR.resolve())


if __name__ == "__main__":
    main()
