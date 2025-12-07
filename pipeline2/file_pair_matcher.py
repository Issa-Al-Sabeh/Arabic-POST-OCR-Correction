#!/usr/bin/env python3
"""
File Pair Matcher and Label Data Creator
Reads from two folders, finds matching files by name, and creates paired label data
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FilePairMatcher:
    """
    Matches files from two folders by name and creates label data
    """
    
    def __init__(self, ocr_folder="arabic_ocr", xml_folder="xml_converted_json"):
        self.ocr_folder = Path(ocr_folder)
        self.xml_folder = Path(xml_folder) 
        self.label_data = []
        
        logger.info(f" OCR folder: {self.ocr_folder}")
        logger.info(f" Ground truth folder: {self.xml_folder}")
    
    def find_matching_files(self) -> List[Tuple[Path, Path]]:
        """
        Find files with matching base names from both folders
        Returns list of (ocr_file, xml_file) tuples
        """
        matches = []
        
        # Get all files from both folders
        ocr_files = list(self.ocr_folder.glob("*.json"))
        xml_files = list(self.xml_folder.glob("*.json"))
        
        logger.info(f" Found {len(ocr_files)} OCR files")
        logger.info(f" Found {len(xml_files)} XML ground truth files")
        
        # Match files by base name
        for ocr_file in ocr_files:
            # Extract base name (e.g., "testing" from "testing.json")
            base_name = ocr_file.stem
            
            # Look for corresponding XML file with same name (e.g., "testing.json")
            xml_file = self.xml_folder / f"{base_name}.json"
            
            if xml_file.exists():
                matches.append((ocr_file, xml_file))
                logger.info(f" Matched: {ocr_file.name} → {xml_file.name}")
            else:
                logger.warning(f" No match found for: {ocr_file.name}")
        
        logger.info(f" Total matches found: {len(matches)}")
        return matches
    
    def create_label_data(self, matches: List[Tuple[Path, Path]]) -> List[Dict]:
        """
        Create label data from matched file pairs
        """
        label_data = []
        
        for ocr_file, xml_file in matches:
            try:
                # Load OCR data
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    ocr_content = json.load(f)
                
                # Load XML ground truth data
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = json.load(f)
                
                # Create label pair
                pair_data = {
                    "pair_id": f"{ocr_file.stem}_{xml_file.stem}",
                    "source_files": {
                        "ocr_file": str(ocr_file),
                        "xml_file": str(xml_file)
                    },
                    "ocr_data": ocr_content,
                    "ground_truth_data": xml_content,
                    "created_from": {
                        "ocr_folder": str(self.ocr_folder),
                        "xml_folder": str(self.xml_folder)
                    }
                }
                
                label_data.append(pair_data)
                logger.info(f" Created label data for: {ocr_file.name} + {xml_file.name}")
                
            except Exception as e:
                logger.error(f" Error processing {ocr_file.name} + {xml_file.name}: {e}")
        
        return label_data
    
    def save_label_data(self, label_data: List[Dict], output_file: str = "labeled_data/label_data.json"):
        """
        Save the paired label data to a JSON file in labeled_data folder
        """
        try:
            # Create labeled_data directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f" Saved {len(label_data)} label pairs to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f" Error saving label data: {e}")
            return False
    
    def get_label_data_summary(self, label_data: List[Dict]) -> Dict:
        """
        Get summary statistics of the label data
        """
        if not label_data:
            return {"total_pairs": 0}
        
        summary = {
            "total_pairs": len(label_data),
            "pair_ids": [pair["pair_id"] for pair in label_data],
            "source_folders": {
                "ocr_folder": str(self.ocr_folder),
                "xml_folder": str(self.xml_folder)
            }
        }
        
        # Count total pages across all pairs
        total_ocr_pages = 0
        total_xml_pages = 0
        
        for pair in label_data:
            ocr_pages = len(pair["ocr_data"].get("pages", []))
            xml_pages = len(pair["ground_truth_data"].get("pages", []))
            total_ocr_pages += ocr_pages
            total_xml_pages += xml_pages
        
        summary["total_pages"] = {
            "ocr_pages": total_ocr_pages,
            "xml_pages": total_xml_pages
        }
        
        return summary
    
    def process_all(self, output_file: str = "labeled_data/label_data.json"):
        """
        Complete workflow: find matches, create label data, and save
        """
        logger.info(" Starting file pair matching and label data creation...")
        
        # Step 1: Find matching files
        matches = self.find_matching_files()
        
        if not matches:
            logger.warning(" No matching files found!")
            return None
        
        # Step 2: Create label data
        label_data = self.create_label_data(matches)
        
        # Step 3: Save to file
        success = self.save_label_data(label_data, output_file)
        
        if success:
            # Step 4: Show summary
            summary = self.get_label_data_summary(label_data)
            
            logger.info(" Label data creation completed!")
            logger.info(f" Summary: {summary}")
            
            return label_data
        
        return None

def main():
    """
    Main function to create label data from file pairs
    """
    # Create the pair matcher
    matcher = FilePairMatcher()
    
    # Process all files and create label data
    label_data = matcher.process_all("labeled_data/label_data.json")
    
    if label_data:
        print(f"\n Successfully created label data with {len(label_data)} pairs!")
        print(f" Saved to: labeled_data/label_data.json")
        
        # Show first pair as example
        if label_data:
            first_pair = label_data[0]
            print(f"\n Example pair: {first_pair['pair_id']}")
            print(f"   OCR file: {first_pair['source_files']['ocr_file']}")
            print(f"   XML file: {first_pair['source_files']['xml_file']}")
    else:
        print(" Failed to create label data")

if __name__ == "__main__":
    main()