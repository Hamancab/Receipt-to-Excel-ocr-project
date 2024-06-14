import csv
import re
import os
import cv2
import easyocr
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
from crap import crap  # Assuming this is a custom module
from prepro import preprocces  # Assuming this is a custom module

image_path = r"Pics\image.png"

preprocces(image_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['tr'])  # 'tr' for Turkish, replace with your language code if needed

# Detect text boxes
result = reader.readtext("input.tiff",
                         slope_ths=0.5,
                         width_ths=1000,
                         link_threshold=0.4,
                         text_threshold=0.7,
                         decoder="wordbeamsearch")

# Initialize an empty list to store bounding boxes
bounding_boxes = []

# Extract bounding boxes from EasyOCR result
for detection in result:
    bounding_boxes.append(detection[0])  # detection[0] contains the bounding box

# Load the image using OpenCV
img = cv2.imread("input.tiff")

# Initialize pytesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
custom_config = r'--oem 1 --psm 13 -c preserve_interword_spaces=1 -l tur --dpi 300'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Perform OCR using Tesseract on each bounding box
csv_filename = 'output.csv'
ocr_output = []  # List to store all OCR results for processing later

with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Bounding Box', 'Detected Text'])  # Write header to CSV file

    for i, bbox in enumerate(bounding_boxes):
        x_min, y_min = [int(min(axis)) for axis in zip(*bbox)]
        x_max, y_max = [int(max(axis)) for axis in zip(*bbox)]
        cropped_region = img[y_min:y_max, x_min:x_max]
        text = pytesseract.image_to_string(cropped_region, config=custom_config)
        ocr_output.append(text)  # Add OCR result to list for later processing
        csv_writer.writerow([f'Bounding Box {i+1}', text])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Display the image with bounding boxes
plt.imsave('bbox.tiff', img)
plt.axis('off')
# plt.show()


# Now, process the OCR results to extract key-value pairs
data_dict = {}
for line in ocr_output:
    # Remove leading and trailing whitespace
    line = line.strip()
    # Split the line into parts
    parts = line.split()
    # Check if the last part is numeric (possibly containing commas)
    if len(parts) >= 2 and re.search(r'^-?\d+(?:[.,]\d+)?$', parts[-1]):
        # Reconstruct the key without the last numeric part
        key = ' '.join(parts[:-1])
        # Take the last part as value
        value = parts[-1].replace(',', '.')  # Convert comma to dot for consistency
        data_dict[key] = value
    # If no numeric part is found at the end, handle specially if a comma is present
    elif ',' in line:
        # Find the last comma in the line
        last_comma_index = line.rfind(',')
        # Find the last space before the last comma (if any)
        last_space_before_comma = line.rfind(' ', 0, last_comma_index)
        # Split into key and value based on the found indices
        if last_space_before_comma != -1:  # A space was found before the comma
            key = line[:last_space_before_comma]
            value = line[last_space_before_comma + 1:].replace(',', '.')  # Convert comma to dot for consistency
        else:
            key = line  # Use the entire line as key if no preceding space
            value = None  # No numeric value to extract
        data_dict[key] = value
    else:  # If the line doesn't fit previous patterns, treat whole line as key
        data_dict[' '.join(parts)] = None

# Write the processed key-value pairs to a new CSV file
with open('processed_output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['column1', 'column2'])  # Header in Turkish: Key, Value
    for key, value in data_dict.items():
        writer.writerow([key, value])

path = r"C:\Users\Hamancab\OCR\processed_output.csv"
df = pd.read_csv(path)
df['column3'] = df['column2'].apply(lambda x: ''.join(c for c in str(x) if c.isdigit() or c in {'.', ','}) if pd.notna(x) else np.nan)
df.to_excel(r"C:\Users\Hamancab\OCR\output.xlsx")

print("Bitti")