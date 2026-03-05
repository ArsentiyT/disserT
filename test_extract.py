#!/usr/bin/env python3
"""
Test script to verify the Russian text extraction functionality.
Creates sample JPG images with Russian text for testing purposes.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_sample_images(num_images=5, output_dir="sample_images"):
    """Create sample JPG images with Russian text for testing."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample Russian text for testing
    russian_texts = [
        "Привет мир! Это тестовое изображение с русским текстом.",
        "OCR-распознавание русского языка требует специальной настройки.",
        "Тестирование точности извлечения текста из изображений.",
        "Система должна корректно распознавать кириллические символы.",
        "Python и Tesseract обеспечивают высокую точность распознавания."
    ]
    
    # Try to find a font that supports Cyrillic characters
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/caladea/Caladea-Regular.ttf"
    ]
    
    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, size=20)
            break
    
    # If no specific font found, use default (might not support Cyrillic well)
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=20)
    
    for i in range(num_images):
        # Create a white image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some random noise to make it more realistic
        pixels = np.array(img)
        noise = np.random.normal(0, 10, pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels)
        
        draw = ImageDraw.Draw(img)
        
        # Draw the Russian text
        text = russian_texts[i % len(russian_texts)]
        draw.text((50, 50), text, fill=(0, 0, 0), font=font)
        
        # Add some additional text at different positions
        additional_text = f"Изображение номер {i+1}"
        draw.text((50, 100), additional_text, fill=(0, 0, 0), font=font)
        
        # Save as JPG
        filename = os.path.join(output_dir, f"test_russian_{i+1:03d}.jpg")
        img.save(filename, "JPEG")
        print(f"Created: {filename}")
    
    print(f"Created {num_images} sample images in {output_dir}/")


if __name__ == "__main__":
    create_sample_images(5, "sample_images")
    print("Sample images created successfully!")
    print("To test the extraction script, run:")
    print("python extract_russian_text.py --input_dir sample_images --output_file extracted_test.txt")