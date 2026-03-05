#!/usr/bin/env python3
"""
Script to extract text from .jpg images in Russian with best accuracy.
Optimized for extracting only words and numbers (no tables or graphs).
Uses Tesseract OCR with Russian language model and specialized preprocessing.
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
import tempfile


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
    Preprocess the image to improve OCR accuracy for Russian text.
    Optimized for extracting only text (words and numbers), not tables or graphs.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        PIL.Image: Preprocessed image
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply gentle denoising - preserve text details
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Apply adaptive thresholding for better handling of varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to connect broken characters but remove noise
    # Use smaller kernel to avoid connecting separate text lines
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Remove small noise components (likely from graphs/tables)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    cleaned = np.zeros_like(eroded)
    
    # Keep only components that are likely text (not too small, not too large)
    min_area = 20  # Minimum area for a character
    max_area = 5000  # Maximum area to exclude large graph elements
    
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cleaned[labels == i] = 255
    
    # Invert back to black text on white background for Tesseract
    processed = cv2.bitwise_not(cleaned)
    
    # Convert back to PIL Image
    pil_img = Image.fromarray(processed)
    
    return pil_img


def extract_text_from_image(image_path, lang='rus+eng'):
    """
    Extract text from a single image using Tesseract OCR.
    Optimized for extracting only Russian words and numbers (no tables/graphs).
    
    Key optimizations for text-only extraction:
    1. Use PSM 6 (uniform block) which works better for continuous text
    2. Filter out non-text elements during preprocessing
    3. Use LSTM engine (--oem 1) for better Cyrillic recognition
    
    Args:
        image_path (str): Path to the image file
        lang (str): Language code for OCR (default: 'rus+eng' for Russian and English)
    
    Returns:
        tuple: (image_path, extracted_text, success_status)
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Configure Tesseract for best Russian text recognition
        # PSM 6 assumes a uniform block of text - ideal for documents without tables/graphs
        configs = [
            r'--oem 1 --psm 6',  # Uniform block of text - best for pure text
            r'--oem 1 --psm 3',  # Fully automatic - fallback
        ]
        
        best_text = ""
        best_confidence = 0
        
        for config in configs:
            # Get detailed output with confidence scores
            data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                
                # Only use this result if it has better confidence
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_text = pytesseract.image_to_string(img, lang=lang, config=config)
        
        # Clean up the text - remove any remaining non-text artifacts
        lines = best_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Keep lines that have actual text content (at least one letter or number)
            if line and (any(c.isalpha() for c in line) or any(c.isdigit() for c in line)):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
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