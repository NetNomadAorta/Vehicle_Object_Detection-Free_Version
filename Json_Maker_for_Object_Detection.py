# Import the necessary packages
import os
import glob
import imutils
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
STICHED_IMAGES_DIRECTORY = "Images/Stitched_Images/"
GOLDEN_IMAGES_DIRECTORY = "Images/Golden_Images/"
SLEEP_TIME = 0.00 # Time to sleep in seconds between each window step
SHOW_WINDOW = False
PRINT_INFO = True
SAVE_WAFERMAP = False
NUMBER_TO_RUN = 50


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


def slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
    # Slides a window across the stitched-image
    for y in range(0, fullImage.shape[0] - windowSize[1]+stepSizeY, stepSizeY):
        for x in range(0, fullImage.shape[1] - windowSize[0]+stepSizeX, stepSizeX):
            if (y + windowSize[1]) > fullImage.shape[0]:
                y = fullImage.shape[0] - windowSize[1]
            if (x + windowSize[0]) > fullImage.shape[1]:
                x = fullImage.shape[1] - windowSize[0]
            # Yield the current window
            yield (x, y, fullImage[y:y + windowSize[1], x:x + windowSize[0]])


# Comparison scan window-image to golden-image
def getMatch(window, goldenImage, x, y):
    h1, w1, c1 = window.shape
    h2, w2, c2 = goldenImage.shape
    
    if c1 == c2 and h2 <= h1 and w2 <= w1:
        method = eval('cv2.TM_CCOEFF_NORMED')
        res = cv2.matchTemplate(window, goldenImage, method)   
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > MATCH_CL: 
            # print("\nFOUND MATCH")
            # print("max_val = ", max_val)
            # print("Window Coordinates: x1:", x + max_loc[0], "y1:", y + max_loc[1], \
            #       "x2:", x + max_loc[0] + w2, "y2:", y + max_loc[1] + h2)
            
            # Gets coordinates of cropped image
            return (max_loc[0], max_loc[1], max_loc[0] + w2, max_loc[1] + h2, max_val)
        
        else:
            return ("null", "null", "null", "null", "null")


# MAIN():
# =============================================================================
# Starting stopwatch to see how long process takes
start_time = time.time()

# Clears some of the screen for asthetics
print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

goldenImagePath = glob.glob(GOLDEN_IMAGES_DIRECTORY + "*")
goldenImage = cv2.imread(goldenImagePath[0])
goldenImage = cv2.rotate(goldenImage, cv2.ROTATE_90_COUNTERCLOCKWISE)



# Parameter set
winW = round(goldenImage.shape[1] * 1.5) # Scales window width with full image resolution
# BELOW DEFAULT IS 1.5 CHANGE BACK IF NEEDED
winH = round(goldenImage.shape[0] * 1.5) # Scales window height with full image resolution
windowSize = (winW, winH)
stepSizeX = round(winW / 2.95)
stepSizeY = round(winH / 2.95)

# Predefine next for loop's parameters 
prev_y1 = stepSizeY * 9 # Number that prevents y = 0 = prev_y1
prev_x1 = stepSizeX * 9
rowNum = 0
colNum = 0
prev_matchedCL = 0


# For Json file
image_names = []
image_ids = []
image_heights = []
image_widths = []

# For Json file
count_below_70 = 0
die_index = -1
die_ids = []
die_image_ids = []
category_id = []
bboxes = np.zeros([1, 4], np.int32)
bbox_areas = []
die_segmentations = []
die_iscrowd = []

for image_index, image_name in enumerate(os.listdir(STICHED_IMAGES_DIRECTORY)):
    
    # TESTING - Only completes up to index (NUMBER_TO_RUN-1)
    if image_index == NUMBER_TO_RUN:
        break
    print("\n\nStarting", image_name, "Image Number:", (image_index+1), "\n")
    # For Json file
    image_names.append(image_name)
    image_ids.append(image_index)
    
    fullImagePath = os.path.join(STICHED_IMAGES_DIRECTORY, image_name)
    
    fullImage = cv2.imread(fullImagePath)
    fullImage = cv2.rotate(fullImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # For Json file
    image_heights.append(fullImage.shape[0])
    image_widths.append(fullImage.shape[1])
    
    cv2.destroyAllWindows()

    # Get's path's actual column and row number
    path_row_number = int(fullImagePath[-18:-16])
    path_col_number = int(fullImagePath[-11:-9])
    if path_row_number % 2 == 1:
        path_row_number = (path_row_number + 1) // 2
    else:
        path_row_number = 20 + path_row_number // 2
    
    if path_col_number % 2 == 1:
        path_col_number = (path_col_number + 1) // 2
    else:
        path_col_number = 20 + path_col_number // 2    
    
    box_count = 0 # Number of boxes made per full 100-die image
    
    # loop over the sliding window
    for (x, y, window) in slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW or box_count==100:
            continue
        
        # Draw rectangle over sliding window for debugging and easier visual
        displayImage = fullImage.copy()
        cv2.rectangle(displayImage, (x, y), (x + winW, y + winH), (255, 0, 180), 2)
        displayImageResize = cv2.resize(displayImage, (850, round(fullImage.shape[0] / fullImage.shape[1] * 850)))
        if SHOW_WINDOW:
            cv2.imshow(str(fullImagePath), displayImageResize) # TOGGLE TO SHOW OR NOT
        cv2.waitKey(1)
        time.sleep(SLEEP_TIME) # sleep time in ms after each window step
        
        # Scans window for matched image
        # ==================================================================================
        # Scans window and grabs cropped image coordinates relative to window
        # Uses each golden image in the file if multiple part types are present
        for goldenImagePath in glob.glob(GOLDEN_IMAGES_DIRECTORY + "*"):
            goldenImage = cv2.imread(goldenImagePath)
            goldenImage = cv2.rotate(goldenImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Gets coordinates relative to window of matched dies within a Stitched-Image
            win_x1, win_y1, win_x2, win_y2, matchedCL = getMatch(window, goldenImage, x, y)
            
            # Saves cropped image and names with coordinates
            if win_x1 != "null":
                # Turns cropped image coordinates relative to window to stitched-image coordinates
                x1      = x + win_x1
                y1      = y + win_y1
                x2      = x + win_x2
                y2      = y + win_y2
                bbox_width   = x2 - x1
                bbox_height  = y2 - y1
                bbox_area    = bbox_width * bbox_height
                
                # Makes sure same image does not get saved as different names
                if (y1 >= (prev_y1 + round(goldenImage.shape[0] * .9) ) 
                    or y1 <= (prev_y1 - round(goldenImage.shape[0] * .9) ) ):
                    rowNum += 1
                    colNum = 1
                    prev_matchedCL = 0
                    sameCol = False
                else:
                    if x1 >= (prev_x1 + round(goldenImage.shape[1] * .9) ):
                        colNum += 1
                        prev_matchedCL = 0
                        sameCol = False
                    else: 
                        sameCol = True
                
                if sameCol == False: 
                    box_count += 1
                    
                    # JSON info
                    die_index += 1
                    die_ids.append(die_index)
                    die_image_ids.append(image_ids[-1]) # image_id to place in annotations category
                    category_id.append(1)
                    if die_index == 0:
                        bboxes[-1] = np.array([x1, y1, bbox_width, bbox_height], ndmin=2)
                    else:
                        bboxes = np.append(bboxes, [[x1, y1, bbox_width, bbox_height]], axis=0)
                    bbox_areas.append(bbox_area)
                    die_segmentations.append([])
                    die_iscrowd.append(0)
                    
                    prev_y1 = y1
                    prev_x1 = x1
                    prev_matchedCL = matchedCL
                    
                    if PRINT_INFO:
                        if (die_index+1) < 10:
                            print("  Die Number:", (die_index+1), "  matchedCL:", round(matchedCL,3))
                        elif (die_index+1) < 100:
                            print("  Die Number:", (die_index+1), " matchedCL:", round(matchedCL,3))
                        else:
                            print("  Die Number:", (die_index+1), "matchedCL:", round(matchedCL,3))
                    else: 
                        if die_index % 100 == 0:
                            print(" ", die_index, "out of", (NUMBER_TO_RUN*100))
                    
                # If the same die is saved twice, and the second one has better coordinates,
                #  then below will replace the last data entry with the better data
                elif sameCol == True and matchedCL > prev_matchedCL:
                    bboxes[-1] = np.array([x1, y1, bbox_width, bbox_height], ndmin=2)
                    bbox_areas[-1] = bbox_area
                    
                    if PRINT_INFO:
                        if (die_index+1) < 10:
                            print("                  matchedCL:", round(matchedCL,3))
                        elif (die_index+1) < 100:
                            print("                  matchedCL:", round(matchedCL,3))
                        else:
                            print("                  matchedCL:", round(matchedCL,3))
                    
                    prev_y1 = y1
                    prev_x1 = x1
                    prev_matchedCL = matchedCL
                
    for i in range(100-box_count):
        # JSON info
        die_index += 1
        die_ids.append(die_index)
        die_image_ids.append(image_ids[-1]) # image_id to place in annotations category
        category_id.append(1)
        bboxes = np.append(bboxes, [[1000, 1000, 180, 180]], axis=0)
        bbox_areas.append(32400)
        die_segmentations.append([])
        die_iscrowd.append(0)
            # ==================================================================================
    rowNum = 0
    colNum = 0
    sameCol = False
    
    if SAVE_WAFERMAP:
        
        
        fullImage_bboxes = fullImage.copy()
        
        for index in die_ids:
            if image_index == die_image_ids[index]:
                cv2.rectangle(fullImage_bboxes, 
                              (int(bboxes[index][0]), int(bboxes[index][1])), 
                              (int(bboxes[index][0] + bboxes[index][2]), 
                               int(bboxes[index][1]) + bboxes[index][3]), 
                              (255, 255, 255), 
                              round(goldenImage.size * 0.00001))
        
        cv2.imwrite("./waferMap-{}.jpg".format(image_index), fullImage_bboxes)



# Creating JSON section
# ==================================================================================
data = {
    "info": {
        "year": "2022",
        "version": "1",
        "description": "Created own",
        "contributor": "Troy P.",
        "url": "",
        "date_created": "2022-02-13T01:11:34+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "die",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "die",
            "supercategory": "die"
        }
    ]
}



# Updates "image" section oc coco.json
for image_index in image_ids:
    if image_index == 0:
        to_update_with = {
            "images": [
                {
                    "id": image_index,
                    "license": 1,
                    "file_name": image_names[image_index],
                    "height": image_heights[image_index],
                    "width": image_widths[image_index],
                    "date_captured": "2022-02-13T01:11:34+00:00"
                }
            ]
        }
        data.update(to_update_with)
    else:
        to_update_with = {
            "images": {
                    "id": image_index,
                    "license": 1,
                    "file_name": image_names[image_index],
                    "height": image_heights[image_index],
                    "width": image_widths[image_index],
                    "date_captured": "2022-02-13T01:11:34+00:00"
            }
        }
        data["images"].append(to_update_with["images"])


# Updates "annotations" section oc coco.json
for die_index in die_ids:
    if die_index == 0:
        to_update_with = {
            "annotations": [
                {
                    "id": die_index,
                    "image_id": die_image_ids[die_index],
                    "category_id": category_id[die_index],
                    "bbox": [
                        int(bboxes[die_index][0]),
                        int(bboxes[die_index][1]),
                        int(bboxes[die_index][2]),
                        int(bboxes[die_index][3])
                    ],
                    "area": bbox_areas[die_index],
                    "segmentation": die_segmentations[die_index],
                    "iscrowd": die_iscrowd[die_index]
                }
            ]
        }
        
        data.update(to_update_with)
        
    else:
        to_update_with = {
            "annotations": {
                    "id": die_index,
                    "image_id": die_image_ids[die_index],
                    "category_id": category_id[die_index],
                    "bbox": [
                        int(bboxes[die_index][0]),
                        int(bboxes[die_index][1]),
                        int(bboxes[die_index][2]),
                        int(bboxes[die_index][3])
                    ],
                    "area": bbox_areas[die_index],
                    "segmentation": die_segmentations[die_index],
                    "iscrowd": die_iscrowd[die_index]
            }
        }
        
        data["annotations"].append(to_update_with["annotations"])


with open('_annotations.coco.json', 'w') as f:
    json.dump(data, f, indent=4)

# ==================================================================================




print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)