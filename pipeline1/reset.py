import os
import shutil
from pathlib import Path

# =============================
# PATHS
# =============================
BASE_DIR = Path(__file__).resolve().parent

INPUT_TXT_DIR = BASE_DIR / "input_txt"
DONE_TXT_DIR = BASE_DIR / "input_txt_done"
OUTPUT_PDF_DIR = BASE_DIR / "output_pdf"
OUTPUT_TXT_DIR = BASE_DIR / "output_txt"   # <--- OCR txt folder

# Ensure directories exist
INPUT_TXT_DIR.mkdir(exist_ok=True)
DONE_TXT_DIR.mkdir(exist_ok=True)
OUTPUT_PDF_DIR.mkdir(exist_ok=True)
OUTPUT_TXT_DIR.mkdir(exist_ok=True)


def reset_output_pdf():
    print("\n🗑️ Deleting PDFs from output_pdf...")

    deleted = 0
    for pdf_file in OUTPUT_PDF_DIR.glob("*.pdf"):
        try:
            pdf_file.unlink()
            deleted += 1
        except Exception as e:
            print(f"  ❌ Could not delete {pdf_file.name}: {e}")

    print(f"  ✅ Deleted {deleted} PDF files.")


def reset_output_ocr_txt():
    print("\n🗑️ Deleting OCR TXT files from output_txt...")

    deleted = 0
    for txt_file in OUTPUT_TXT_DIR.glob("*.txt"):
        try:
            txt_file.unlink()
            deleted += 1
        except Exception as e:
            print(f"  ❌ Could not delete {txt_file.name}: {e}")

    print(f"  ✅ Deleted {deleted} OCR TXT files.")


def reset_txt_files():
    print("\n📦 Moving TXT files from input_txt_done ➝ input_txt...")

    moved = 0
    for txt_file in DONE_TXT_DIR.glob("*.txt"):
        dest = INPUT_TXT_DIR / txt_file.name
        try:
            shutil.move(str(txt_file), str(dest))
            moved += 1
        except Exception as e:
            print(f"  ❌ Could not move {txt_file.name}: {e}")

    print(f"  ✅ Moved {moved} TXT files.")


def main():
    print("🔄 RESET PIPELINE STARTED")

    reset_output_pdf()
    reset_output_ocr_txt()
    reset_txt_files()

    print("\n🎉 RESET COMPLETE!")


if __name__ == "__main__":
    main()
