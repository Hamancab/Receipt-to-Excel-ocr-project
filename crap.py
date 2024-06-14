import cv2
import numpy as np
import imutils
import os
import logging
import time

def crap(path):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Your image path
    image_path = path
    image = cv2.imread(image_path)

    def convert_hsv(image):
        # Your implementation for converting to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv_image

    def mask_img(hsv_img, lower, upper):
        # Your implementation for masking
        mask = cv2.inRange(hsv_img, lower, upper)
        return mask

    def generate_contours_filename():
        # Your implementation for generating a filename for contours
        # You might want to generate a unique filename based on the current time or other criteria
        timestamp = str(int(time.time()))
        return f"contours_{timestamp}.png"

    def generate_filename():
        # Your implementation for generating a filename
        # You might want to generate a unique filename based on the current time or other criteria
        timestamp = str(int(time.time()))
        return f"cropped_{timestamp}.png"

    def crop_image(image):
        original_img = image.copy()
        hsv_img = convert_hsv(image)

        lower_blue = np.array([0, 0, 120])
        upper_blue = np.array([180, 38, 255])

        masked_image = mask_img(hsv_img, lower_blue, upper_blue)
        result = cv2.bitwise_and(image, image, mask=masked_image)
        contours = cv2.findContours(masked_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        cv2.drawContours(masked_image, contours, -1, (0, 255, 0), 3)
        max_area_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_area_contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # cont_filename = generate_contours_filename()
        # cv2.imwrite(cont_filename, np.hstack([image, result]))
        # logger.info('Successfully saved file: %s' % cont_filename)
        
        img = image[y:y+h, x:x+w]
        # filename = generate_filename()
        # cv2.imwrite(filename, img)
        # logger.info('Successfully saved cropped file: %s' % filename)
        
        return img#, filename



    # Call your crop_image function
    cropped_image = crop_image(image)

    # Print the filename of the cropped image
    # print("Cropped Image Filename:", os.path.basename(cropped_filename))

    # cv2.imshow("Cropped Image", cropped_image)
    # cv2.waitKey(0)  # Bekleme süresi, sıfır kullanıldığında kullanıcı bir tuşa basana kadar bekler

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
    return cropped_image