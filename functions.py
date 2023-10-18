import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_image(image):
    equalized = cv2.equalizeHist(image)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def morph_ops(image, kernel):
    # Apply binary threshold to the image
    _, gray_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    # Apply morphological operations on the binary image
    closed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    opened_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    eroded_image = cv2.erode(gray_image, kernel, iterations=1)
    dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

    return closed_image, opened_image, eroded_image, dilated_image