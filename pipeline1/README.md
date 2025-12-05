# Arabic OCR Pipeline

## 📁 Folder Structure

### **input_txt/**

Contains the original `.txt` files that the pipeline will process.

### **input_txt_done/**

After a `.txt` file is processed, it is moved here.  
This prevents re-processing the same file.

### **output_pdf/**

Stores the final noisy/scanned-like PDF generated from each `.txt` file.

### **output_json/**

Stores the OCR output **JSON** generated from the noisy PDFs.  
Each JSON file has the same base name as its source `.txt` / PDF.

### **temp_pdf/**

Temporary folder used to store clean PDFs before noise is applied.  
Files here are deleted automatically during processing.

---

## 📄 Files

### **Amiri-Regular.ttf**

The font used to render Arabic text when converting `.txt` files to PDF.

### **pipeline.py**

The main pipeline script. It:

1. Converts `.txt` → **clean PDF**
2. Adds noise → **noisy scanned-like PDF** (saved in `output_pdf/`)
3. Runs OCR on the noisy PDF and builds a **custom JSON structure**
4. Saves the OCR JSON in `output_json/`
5. Moves the processed `.txt` file to `input_txt_done/`

### **reset.py**

Resets the project by:

- Deleting all generated PDFs from `output_pdf/`
- Deleting all OCR JSON files from `output_json/`
- Moving processed `.txt` files back from `input_txt_done/` to `input_txt/`

---

## ✔️ Usage

Place your `.txt` files inside **input_txt/**, then run:

python pipeline.py

To reset the project to its initial state:

python reset.py

---
