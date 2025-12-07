#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import json
import re
from pathlib import Path
import os

# ===========================
# SETTINGS
# ===========================
XML_INPUT_DIR = r"C:\Users\ali-d\Desktop\output"  # Folder containing XML files
JSON_OUTPUT_DIR = "xml_converted_json"  # Output folder inside pipeline2

def convert_xml_to_json(xml_file_path, output_json_path):
    """
    Convert ABBYY FineReader XML to the same JSON format as tesseract OCR output
    Supports multiple pages within a single XML file
    """
    print(f"Converting: {Path(xml_file_path).name}")
    
    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Define namespace
        ns = {'abbyy': 'http://www.abbyy.com/FineReader_xml/FineReader10-schema-v1.xml'}
        
        # Get document name from file path
        document_name = Path(xml_file_path).name
        
        # Initialize result structure
        ocr_result = {
            "document": document_name,
            "total_pages": 0,
            "pages": []
        }
        
        # Find all pages
        pages = root.findall('.//abbyy:page', ns)
        if not pages:
            pages = root.findall('.//page')  # Try without namespace
        
        if not pages:
            print(f"   No pages found in {document_name}")
            return
        
        for page_num, page in enumerate(pages, start=1):
            print(f"  - Processing page {page_num}...", end=" ")
            
            # Get page dimensions
            page_width = int(page.get('width', 0))
            page_height = int(page.get('height', 0))
            
            # Collect all text and word data
            words = []
            all_text_parts = []
            
            # Find all text blocks
            text_blocks = page.findall('.//text') if not ns else page.findall('.//abbyy:text', ns)
            if not text_blocks:
                text_blocks = page.findall('.//text')
            
            word_num_global = 1
            block_num = 1
            
            for block in text_blocks:
                line_num = 1
                
                # Find all lines in this block
                lines = block.findall('.//line') if not ns else block.findall('.//abbyy:line', ns)
                if not lines:
                    lines = block.findall('.//line')
                
                for line in lines:
                    word_num_line = 1
                    
                    # Find all formatting elements (words) in this line
                    formatting_elements = line.findall('.//formatting') if not ns else line.findall('.//abbyy:formatting', ns)
                    if not formatting_elements:
                        formatting_elements = line.findall('.//formatting')
                    
                    for formatting in formatting_elements:
                        text_content = formatting.text if formatting.text else ""
                        if text_content.strip():
                            all_text_parts.append(text_content)
                            
                            # Get line coordinates
                            line_l = int(line.get('l', 0))
                            line_t = int(line.get('t', 0))
                            line_r = int(line.get('r', 0))
                            line_b = int(line.get('b', 0))
                            
                            # Calculate approximate word position within line
                            line_width = line_r - line_l
                            line_height = line_b - line_t
                            
                            # Split text content into individual words
                            individual_words = text_content.split()
                            for word_idx, word in enumerate(individual_words):
                                if word.strip():
                                    # Estimate word position within the line
                                    word_ratio = (word_idx + 0.5) / len(individual_words) if len(individual_words) > 1 else 0.5
                                    estimated_x = line_l + int(line_width * word_ratio * 0.8)  # Approximate position
                                    estimated_width = max(len(word) * 8, 20)  # Rough estimate based on character count
                                    
                                    word_info = {
                                        "text": word,
                                        "confidence": 85,  # Default confidence for ABBYY XML
                                        "bbox": {
                                            "x": estimated_x,
                                            "y": line_t,
                                            "width": estimated_width,
                                            "height": line_height
                                        },
                                        "level": 5,  # Word level
                                        "block_num": block_num,
                                        "line_num": line_num,
                                        "word_num": word_num_line
                                    }
                                    words.append(word_info)
                                    word_num_line += 1
                                    word_num_global += 1
                    
                    line_num += 1
                block_num += 1
            
            # Join all text parts for this page
            page_text = "\n".join(all_text_parts)
            
            # Count actual words (non-empty)
            word_count = len([w for w in words if w['text'].strip()])
            
            page_data = {
                "page_number": page_num,
                "text": page_text.strip(),
                "word_count": word_count,
                "words": words
            }
            
            ocr_result["pages"].append(page_data)
            print(f"done ({word_count} words)")
        
        # Update total pages
        ocr_result["total_pages"] = len(ocr_result["pages"])
        
        # Save as JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)
        
        total_words = sum(page['word_count'] for page in ocr_result['pages'])
        print(f"   Converted to JSON: {Path(output_json_path).name}")
        print(f"   Pages: {ocr_result['total_pages']}, Total words: {total_words}\n")
        
    except Exception as e:
        print(f"   Error converting {Path(xml_file_path).name}: {e}\n")


def main():
    """
    Process all XML files in the input directory and convert them to JSON
    """
    # Setup directories
    xml_input_path = Path(XML_INPUT_DIR)
    
    # Create output directory inside pipeline2
    current_dir = Path(__file__).parent  # pipeline2 directory
    json_output_path = current_dir / JSON_OUTPUT_DIR
    json_output_path.mkdir(exist_ok=True)
    
    # Find all XML files in input directory
    xml_files = list(xml_input_path.glob("*.xml"))
    
    if not xml_files:
        print(f" No XML files found in {xml_input_path}")
        print(f" Make sure XML files are in: {xml_input_path.resolve()}")
        return
    
    print(f" Found {len(xml_files)} XML files in {xml_input_path}")
    print(f" Output directory: {json_output_path.resolve()}")
    print(f"{'='*60}")
    
    # Process each XML file (skip if JSON already exists)
    successful_conversions = 0
    skipped = 0
    for xml_file in xml_files:
        # Generate output JSON filename
        json_filename = xml_file.stem + ".json"
        json_file_path = json_output_path / json_filename

        # Skip if already converted
        if json_file_path.exists():
            print(f"[SKIP] Already converted: {json_filename}")
            skipped += 1
            continue
        
        # Convert XML to JSON
        convert_xml_to_json(xml_file, json_file_path)
        
        if json_file_path.exists():
            successful_conversions += 1
    
    print(f"{'='*60}")
    print(f" Conversion complete!")
    print(f" Successfully converted: {successful_conversions}/{len(xml_files)} files")
    print(f"⏭  Skipped existing: {skipped} files")
    print(f" JSON files saved in: {json_output_path.resolve()}")


if __name__ == "__main__":
    main()