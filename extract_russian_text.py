#!/usr/bin/env python3
"""
Script to extract text from .jpg images in Russian with best accuracy.
Uses Tesseract OCR with Russian language model and various preprocessing techniques.
"""

import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import argparse
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ocr_extraction.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR accuracy.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        PIL.Image: Preprocessed image
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to enhance text
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to PIL Image
    pil_img = Image.fromarray(processed)
    
    return pil_img


def extract_text_from_image(image_path, lang='rus'):
    """
    Extract text from a single image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file
        lang (str): Language code for OCR (default: 'rus' for Russian)
    
    Returns:
        tuple: (image_path, extracted_text, success_status)
    """
    try:
        # Preprocess the image
        preprocessed_img = preprocess_image(image_path)
        
        # Configure Tesseract for better accuracy
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 !\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"'
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(preprocessed_img, lang=lang, config=custom_config)
        
        # Clean up the text
        text = text.strip()
        
        return str(image_path), text, True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return str(image_path), "", False


def validate_inputs(input_dir, output_file):
    """Validate input parameters."""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    input_path = Path(input_dir)
    jpg_files = list(input_path.glob("*.jpg"))
    
    if not jpg_files:
        raise ValueError(f"No .jpg files found in directory: {input_dir}")
    
    print(f"Found {len(jpg_files)} .jpg files in {input_dir}")
    
    return jpg_files


def save_results(results, output_file):
    """
    Save extracted text results to a file.
    
    Args:
        results (list): List of tuples containing (image_path, text, success_status)
        output_file (str): Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        successful_count = 0
        failed_count = 0
        
        for image_path, text, success in results:
            f.write(f"--- Image: {image_path} ---\n")
            if success:
                f.write(text)
                successful_count += 1
            else:
                f.write("[FAILED TO EXTRACT TEXT]")
                failed_count += 1
            f.write("\n\n")
        
        f.write(f"\nSummary:\n")
        f.write(f"Total images processed: {len(results)}\n")
        f.write(f"Successfully processed: {successful_count}\n")
        f.write(f"Failed to process: {failed_count}\n")


def main():
    global logger
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Extract text from Russian .jpg images using OCR")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing .jpg images")
    parser.add_argument("--output_file", type=str, default="extracted_text.txt",
                        help="Output file to save extracted text (default: extracted_text.txt)")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of threads for parallel processing (default: 4)")
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        jpg_files = validate_inputs(args.input_dir, args.output_file)
        
        # Limit to 150 files if more exist
        if len(jpg_files) > 150:
            jpg_files = jpg_files[:150]
            logger.info(f"Limited to first 150 files out of {len(jpg_files)} total")
        
        logger.info(f"Processing {len(jpg_files)} images...")
        
        # Process images in parallel
        results = []
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(extract_text_from_image, str(img_path)): str(img_path) 
                for img_path in jpg_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                result = future.result()
                results.append(result)
                
                # Log progress
                completed = len([r for r in results if r[2]])  # Count successful
                logger.info(f"Processed {completed}/{len(jpg_files)} images successfully")
        
        # Sort results by image path for consistent output
        results.sort(key=lambda x: x[0])
        
        # Save results
        save_results(results, args.output_file)
        
        logger.info(f"Text extraction completed! Results saved to {args.output_file}")
        
        # Print summary
        successful = sum(1 for _, _, success in results if success)
        print(f"\nExtraction Summary:")
        print(f"Total images processed: {len(results)}")
        print(f"Successful extractions: {successful}")
        print(f"Failed extractions: {len(results) - successful}")
        print(f"Results saved to: {args.output_file}")
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()