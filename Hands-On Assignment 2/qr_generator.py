"""

QR Code Generator
Name: Rabi gurung
Assignment: Hands-On Assignment 2 - Construct AI QR Code Generator with Python
Course: MSCS-633-M50 (Advanced Artificial Intelligence)

This application generates QR codes from URL addresses.
"""

import qrcode
import os
from pathlib import Path


def validate_url(url):
    """Validate URL format."""
    url = url.strip()
    valid_schemes = ('http://', 'https://', 'ftp://', 'www.')
    return any(url.lower().startswith(scheme) for scheme in valid_schemes)


def generate_qr_code(url, output_path='output/qr_code.png'):
    """
    Generate a QR code from a given URL.
    
    Args:
        url: The URL to encode
        output_path: Path to save the QR code image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not validate_url(url):
            print("Error: Invalid URL format.")
            return False
        
        Path(os.path.dirname(output_path) or '.').mkdir(parents=True, exist_ok=True)
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(output_path)
        
        print(f"QR code generated: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def main():
    """Main application loop."""
    print("QR Code Generator")
    print("-" * 40)
    
    while True:
        url = input("Enter URL (or 'quit' to exit): ").strip()
        
        if url.lower() == 'quit':
            break
        
        if not url:
            print("Please enter a valid URL.\n")
            continue
        
        if generate_qr_code(url):
            print("Done!\n")
        else:
            print("Try again.\n")


if __name__ == "__main__":
    main()
