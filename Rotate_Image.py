# Import the necessary packages
import os
import cv2
import time
# TESTING SVD FROM NUMPY
import numpy as np
import math
import json


# User Parameters/Constants to Set
MATCH_CL = 0.65 # Minimum confidence level (CL) required to match golden-image to scanned image
# STICHED_IMAGES_DIRECTORY = "//mcrtp-sftp-01/aoitool/LED-Test/Slot_01/"
# GOLDEN_IMAGES_DIRECTORY = "C:/Users/ait.lab/.spyder-py3/Automated_AOI/Golden_Images/"
IMAGES_TO_FLIP_DIRECTORY = "Images/To_Flip_Images/To_Flip_Images/"
SAVE_IMAGES_DIRECTORY = "Images/To_Flip_Images/Flipped_Images/"


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )



# MAIN():
# =============================================================================
# Starting stopwatch to see how long process takes
start_time = time.time()

# Clears some of the screen for asthetics
print("\n\n\n\n\n\n\n\n\n\n\n\n\n")


for image_index, image_name in enumerate(os.listdir(IMAGES_TO_FLIP_DIRECTORY)):
    
    fullImagePath = os.path.join(IMAGES_TO_FLIP_DIRECTORY, image_name)
    
    fullImage = cv2.imread(fullImagePath)
    fullImage = cv2.rotate(fullImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    cv2.imwrite(SAVE_IMAGES_DIRECTORY + image_name, fullImage)


# ==================================================================================




print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)