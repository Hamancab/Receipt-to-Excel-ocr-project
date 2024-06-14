import os
import csv
import cv2
import pytesseract
from PIL import Image
from crap import crap
import numpy as np
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import rotate
from deskew import determine_skew
from openpyxl import Workbook


def preprocces(image_path):

    image = crap(image_path)

    # Resize the image
    # Check for original image dimensions and adjust fx and fy accordingly
    if image.shape[1] > 1000:
        fx = 2  # Adjust for larger images
    else:
        fx = 3  # Maintain for smaller images
    image = cv2.resize(image, None, fx=fx, fy=fx, interpolation=cv2.INTER_LANCZOS4)


    # Deskewing the image
    if image.shape[2] == 4:
        image = rgba2rgb(image)

    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)

    rotated = rotate(image, angle, resize=True) * 255
    rotated = rotated.astype(np.uint8)


    # Convert to grayscale and apply adaptive thresholding
    gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
    median = cv2.medianBlur(blur,5)

    # divide
    divide = cv2.divide(gray, median, scale=256)

    thresh = cv2.adaptiveThreshold(divide, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, blockSize=11, C=11)

    # Denoise and enhance text
    # kalınlaştıran
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    #incelten
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

    erosion = cv2.erode(thresh,kernel,iterations = 2)

    inverted_image = cv2.bitwise_not(erosion)

    # Save the preprocessed image as a TIFF file with DPI information
    tiff_image = Image.fromarray(inverted_image)
    dpi_value = 300
    tiff_image.info['dpi'] = (dpi_value, dpi_value)
    image = tiff_image.save('input.tiff')
    return 
