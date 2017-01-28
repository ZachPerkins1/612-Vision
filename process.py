# -*- coding: utf-8 -*-

# Begin with importing libraries
# cv2 for OpenCV2 interface through python
# os for shell commands that change camera settings
# math for square roots
# socket to get the network information necessary to connect to the NetworkTables server
# numpy for additional math manipulation that is not in the math package
# networktables is the server to send values to so that they are available on the RoboRIO.
#   The networktables server runs on the RoboRIO. 
# logging for DEBUG-level logging
import cv2, os, math, socket
import numpy as np
from networktables import *
import logging
logging.basicConfig(level=logging.DEBUG)

#Intitialize the network table where all the values will be stored
vision = None

# A list of tracking IDs for each target. As new targets are in the camera field of view, they are each given an integer ID, 
# starting from 0. Once a target is out of camera view, OpenCV loses its ID.  The value keeps incrementing (never repeats) 
#as the program runs for a longer time and targets enter field of view and depart the field of view.
ids = [] 

# A list of lists of points representing the paths that each target has followed over MAX_PATH_LEN frames.
# This is represented as an array of arrays of arrays, I.E. Each path is stored as an array, and each point
# inside said path is also stored as an array. Used for interpolation. Synced up with ids such that paths[0]
# represents the same box that ids[0] does.
# paths is used to draw circles of where the target has moved over the last few frames of the video, which
# is convenient for humans to understand the motion of the robot relative to the target.
paths = []

# The current ID (used so that the same ID may not be repeated twice)
curr_id = 0

# The image (pulled from the camera using OpenCV)
# This variable represents all of the pixel values coming off the camera.
# The word frame represents a single image from a stream of images from a camera.  So frame and image are interchangeable. 
# Although we don't usually need to refer to the values directly, some information on the image:
# There are usually 3 indexes: a row, a column, and the color channel (usually 3 -- RGB: red, green, and blue; or HSV: hue, saturation and value).
# The row index starts at 0 being the upper left, and increments left to right in the image.
# the column index starts at 0 being the upper left, and increments top to bottom in the image (note: potentially counter-intuitive).
# the channel is usually from 0 to 2, and the color order can vary depending on the circumstance
image = None

# The MAX_DISTANCE_NEW value is used to track new targets coming within the view of the camera
# In order to ensure the tracking IDs stay the same for the same target, we give a larger value here to allow for some uncertainty
# that occurs from how the target enters the field of view.
# Maximum difference between the interpolated target and the actual target IF the actual target
# doesn't have a previous path (we can't interpolate the value)
MAX_DISTANCE_NEW = 200

# The MAX_DISTANCE_INTERPOLATED value is used to track targets that have been within the camera field of view for 2 or more frames.
# Maximum difference between the interpolated target and the actual target IF the actual target has
# at least 2 frames to interpolate from
MAX_DISTANCE_INTERPOLATED = 100

# Maximum number of points stored in each path, based on the last MAX_PATH_LEN frames from the camera.
# We are using 7 because more than that and the view becomes confusing/too busy to be useful.
MAX_PATH_LEN = 7

# The minimum perimeter of each box for it to be considered valid
# This prevents us from tracking too small of objects or vision targets that are too distant.  
# As smaller perimeters are allowed, we will also make mistakes and track targets that do not really exist
# (are not real field targets, just artifacts from what the camera happens to be looking at).
MIN_PERIMETER = 50

# The amount of times that a target must be absent for it's ID and path to be removed from the list
# This gives us some time lag before we lose track of a target, which is helpful for sudden motions
# on the robot, but that settle quickly.
TOLERANCE = 3

# values.txt represents a list of HSV values that are saved each shutdown so that the sliders
# stay the same between different runs of the program. Helps to adjust color masking.
values1 = open("values.txt", "r")

# The nothing function is used as a placeholder for calling into OpenCV when it expects to do a specific operation.
# Since the OpenCV function call requires a parameter, we provide it a dummy function that does nothing.
def nothing(x): #This is apparently necessary, so yeah
    pass # a python keyword that takes no action

# The main function is where the execution begins in the script
def main():
    global vision # the global keyword inside a function means that the global variable value should be used,
                  # rather than a local variable that happens to have the same name.
    # The RoboRIO uses the mDNS protocol to provide an easier mechanism to find the device on the network
    # Resolve the IP address of the roborio using it's mDNS address
    data = socket.gethostbyname_ex("roborio-612-frc.local")  # data has many values in it
    ip = data[2][0]  # the IP address is the only value we need from data
    
    # Initializes NetworkTables to the IP of the RoboRIO
    print(ip) #Makes sure it's valid, debugging
    NetworkTable.setIPAddress(ip)
    NetworkTable.setClientMode()
    NetworkTable.initialize()
    vision = NetworkTable.getTable("Vision") # the vision variable can now access any key/value pairs from the "Vision" table
    
    # Some NetworkTables testing stuffs 
    # vision.putNumber("test", 2123)  # write the key "test" with value 2123
    # print(vision.getNumber("test", 0))  # confirm that the key "test" has the value we just wrote
    
    global image
    # image = cv2.imread("RealFullField/80.jpg", 1)

    # cap is the capture device of the image.  This means that it is the variable that is used to get new images
    # from the camera.  The camera has a physical device identifier of /dev/video0 on the computer filesystem,
    # so we only need to provide the 0 index in case there were multiple cameras on the computer (then it would be 0, 1, 2, etc.)
    # Set the device of the camera to dev/video0 (*should* be the id of the camera on the Jetson)
    cap = cv2.VideoCapture(0)

    # There isn't a python library that gives direct video setting access, so we use some commands from the command prompt
    # to modify the camera behavior settings
    # Turns off auto exposure
    os.system("uvcdynctrl --device=video0 -s 'Exposure, Auto' 1")
    # Sets camera exposure to be dark (such that the green stands out)
    os.system("uvcdynctrl --device=video0 -s 'Exposure (Absolute)' 9")

    # OpenCV has a build in video display to show the camera feed visually on the Jetson desktop
    # Create a new window to display the video feed
    cv2.namedWindow("image")

    # These values are used to filter the camera image so that H is between Hmin and Hmax, S is between Smin and Smax, and 
    # V is between Vmin and Vmax.  If the HSV value falls outside of any of the ranges, then it is set to 0 (a black pixel).
    # This is how we can filter for the strong green LED that is reflecting back into the camera from the retroreflective tape. 
    # 
    # List of properties for the sliders
    names = ["H Min", "S Min", "V Min", "H Max", "S Max", "V Max"]

    # Create an easy sliderbar to choose the values for the above values while the program is running (rather than hard coded
    # in python)
    # Create new sliders on the image window defaulting to the values from the file
    for index, val in enumerate(values1.readlines()):
        cv2.createTrackbar(names[index], "image", int(val), 255, nothing)

    values1.close()
    # cv2.createTrackbar("Min Size", "image", 0, 10000, nothing)
    # cv2.createTrackbar("Max Size", "image", 0, 10000, nothing)

    # Keep capturing frames... forever...
    while True:
        # Get a frame from the camera
        # ret is a True or False value if the camera read was successful, and unsured in our code
        # by writing to image here, we are writing to the global variable because of the global image above
        ret, image = cap.read()

        # Get the values from the sliders and adjust the filter values properly
        # We call these green because the LED is green.  If the LEDs were a different color, then we'd change that here.
        lower_green = np.array([cv2.getTrackbarPos('H Min', 'image'), cv2.getTrackbarPos('S Min', 'image'),
                                cv2.getTrackbarPos('V Min', 'image')]) #HSV Value
        upper_green = np.array([cv2.getTrackbarPos('H Max', 'image'), cv2.getTrackbarPos('S Max', 'image'),
                                cv2.getTrackbarPos('V Max', 'image')]) #HSV Value

        # lower_green = np.array([83, 35, 157]) should be similar to above
        # upper_green = np.array([153, 166, 224])

        # Blur to reduce noise
        # The blur makes very small pixel regions of 5 by 5 "fuzz" into 0.  It removes spurious small regions that match the
        # HSV color range, which happens fairly often.  If we did not blur them away, the tracking algorithm may assign an ID
        # the small region, which we know is not a real vision target.
        # The actual values of the 5 by 5 that are used to filter are selected to follow the Gaussian distribution (a hill at the
        # center of the 5x5 (so at 3x3), that tapers off into 0 on all sides) 
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Convert Image to HSV so we can filter better
        # The original image is not in HSV format, but in RGB  (BGR is just a shifted ordering)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Mask the image to only include the colors we're looking for
        # A Mask is going to make a monochrome image (rather than RGB it is grayscale) where the pixel value is kept if within the range
        # and set to 0 if it out of the range.  This means that each pixel is 8 bits
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Threshold the image for the sake of all of the other things we have to do
        # The Thresholded image is now a binary mask, of 0 or 1 (still stored as 8 bits for memory purposes), so a pixel is either on or off.
        # the 1 is assigned for any pixel in range of 127 to 255, and assigned a 0 for values 0 to 126.
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Kernel is a matrix of 5x5 with all set to 1.  This is used as a way to remove small pixel regions in the thresh image
        # Each matrix entry is stored as an unsigned integer 8 bits (an unsigned character in C).  This data size was chosen to match
        # the thresh data size so that it uses less math computation than 32 bit values.
        kernel = np.ones((5, 5), np.uint8)

        # Eliminates small, meaningless, peasant edges
        thresh = cv2.erode(thresh, kernel, iterations=1)

        # Combines parts of polygons (allows us to see the target, even with blockage)
        # Dilate tries to grow a region so that neighboring regions may lump together into a single region.
        # If the two or more regions are too far apart in pixel size (beyond iterations=3), then the regions stay separated.
        thresh = cv2.dilate(thresh, kernel, iterations=3)

        # Find the contours (outlines) of all the shapes in the threshold
        # Contours are the bounding boxes along the edges of where the vision target is surrounded by 0 pixels
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Reduce the amount of points in each set of contours
        # We want to draw simplified bounding boxes around each of the vision targets, so we overwrite a contour with fewer data points
        for x in range(0, len(contours)):
            epsilon = 0.01 * cv2.arcLength(contours[x], True)
            contours[x] = cv2.approxPolyDP(contours[x], epsilon, True)

        # Eliminate contours by perimeter
        # This removes entries in contours that are too small, less than MIN_PERIMETER in bounding box length
        contours = [x for x in contours if not cv2.arcLength(x, True) < MIN_PERIMETER]

        # The bounding boxes of the current frame only
        bounding_boxes = []

        # Actually create the bounding boxes and draw them on screen
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)  # OpenCV determines the actual bounding box coordinates from the contour
            bounding_boxes.append([x, y, w, h])  # We store it in an array, x, y are the upper left, and w, h are the width, height
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2) # draw in the image a rectangle (x+w, y+h) is the lower right corner
                                                             # (0, 255, 0) is BGR to draw in green, 2 is the line thickness
        
        # We choose to draw the center point of the last few frames of the same tracked object on the screen, as a circle of 
        # different sizes as the data gets older.  This is just a way to represent the motion of the target over the last few frames 
        # on the computer screen.
        # Go over every path and draw circles for a visual representation. Radius is based on path index.    
        for path in paths:
            for i,p in enumerate(path):
                cv2.circle(image, (int(p[0]),int(p[1])), i, (255,0,0), thickness=-1)
            # hull = cv2.convexHull(c)
            # hulls.append(hull)
            # cv2.drawContours(image, [hull], 0, (0, 255, 0), 3)
        
        # Verify which bounding box is which from frame to frame
        track(bounding_boxes)

        # Display the windows on the Jetson desktop GUI
        cv2.imshow("image", image)
        cv2.imshow("mask", mask)
        cv2.imshow("thresh", thresh)

        # Wait for the escape key and exit, otherwise we loop forever
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    # Write the values of the current slider positions to the file
    # This gives us flexibility to remember the slider settings, which vary between different arenas because of the lighting conditions
    values2 = open("values.txt", "w")
    for name in names:
        values2.write(str(cv2.getTrackbarPos(name, 'image')) + "\n")

    # Close all windows, cleanly end the program
    values2.close()
    cv2.destroyAllWindows()

# !Zach should do a better job of explaining the track function
def track(boxes):
    # access the image and vision without passing to the function call as a parameter (faster)
    global image
    global vision

    allMatched = (len(boxes) == 0 and len(paths) == 0)
    
    if (not allMatched):
        numMatched = 0
        matchedBoxes = []
        
        for x,path in enumerate(paths):
            # Where it guesses it will be next frame given the path over the last few frames
            possibleLocations = interpolateNewPos(path)
            
            # Possible matched boxes. Algorithm works by comparing the interpolated box to the actual box
            # and adds it if it is within MAX_DISTANCE_INTERPOLATED pixels 
            possibleResults = []
            for i,box in enumerate(boxes):
                bx = box[0] + box[2]/2.0 # Center x coordinate
                by = box[1] + box[3]/2.0 # Center y coordinate
                px = possibleLocations[0]
                py = possibleLocations[1]
                dx = abs(bx - px) # Distance between where it is and where it thinks it will be
                dy = abs(by - py)
                
                # Distance formula, now with more degrees of exponential!
                dist = math.sqrt(dx**2+ dy**2) 
                
                # If we have nothing to go off of for interpolation, let the maximum distance be higher.
                max_dist = MAX_DISTANCE_NEW if len(path) < 2 else MAX_DISTANCE_INTERPOLATED 
                
                if dist <= max_dist: #Do nothing if it isn't
                    # Checks to make sure that the result found hasn't already been added to the list of matches
                    found = False
                    for bo in matchedBoxes:
                        if bo == i:
                            found = True
                    if not found:
                        # When a match is found, store its index in the list "boxes", its distance from the 
                        # interpolated target, and it's center coordinate.
                        possibleResults.append([i, dist, bx, by])
            
            # Proceeds to further eliminate if there is more than one box by which one is the closest            
            if len(possibleResults) > 0: # There are possible results
                distances = [i[1] for i in possibleResults] # All the distances for the possible results
                best = distances.index(min(distances)) # Index of the possibleResult with the smallest distance
                tehBestOne = possibleResults[best] # the one with minimum miss distance
                
                # Append the index of the best box in "boxes" to the list of matched boxes
                matchedBoxes.append(tehBestOne[0])
                
                # Add the center coordinates of the object to it's path
                appendToPath(path, [tehBestOne[2], tehBestOne[3]])
                numMatched += 1
                
                #Labels it on screen to make things all pretty-like
                cv2.putText(image, str(ids[x]), (int(tehBestOne[2]), int(tehBestOne[3] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255)) #Labels the value with the ID
            
            # If the box isn't found, we can take one of two paths    
            else:
                # If the final TOLERANCE number of coordinates in the list are equal to -1, -1, we eliminate the box.
                # Because -1 would never occur as an actual coordinate, this allows us to only delete the box if WE
                # have set the coordinate ourselves. This allows us to implement a tolerance system, wherein the box
                # will only dissappear if it is absent for TOLERANCE amount of frames.
                if (len(path) >= TOLERANCE) and (sum([1 for num in path[-TOLERANCE:len(path)] if sum(num) == -2]) == TOLERANCE): 
                    del paths[x]
                    del ids[x]
                else:
                    # If we haven't reached our max tolerance, then add another -1, -1
                    appendToPath(path, [-1, -1])
        
        # Temporary code to communicate with the robot            
        id_arr = NumberArray.from_list(ids)
        vision.putValue("IDS", id_arr)
        
        bounding_arr = NumberArray()
        for box in matchedBoxes:
            for i in range(0, len(boxes[box])):
                bounding_arr.append(boxes[box][i])
           
            
        vision.putValue("BOUNDING_COORDINATES", bounding_arr)
        
       
        
        # If there are extra, unmatched boxes
        if numMatched < len(boxes):
            # Delete all the found ones from the list of boxes so we don't add ones we've already found
            for index in sorted(matchedBoxes, reverse=True):
                del boxes[index]
            
            # Give each new box a tracking ID and a path
            for box in boxes:
                ids.append(newTrackingID())
                cx = box[0] + box[2]/2.0
                cy = box[1] + box[3]/2.0
                paths.append([[cx, cy]])
                
    else:
        # mhm
        print r'''
                      _____ 
                   ,-'     `._ 
                 ,'           `.        ,-. 
               ,'               \       ),.\ 
     ,.       /                  \     /(  \; 
    /'\\     ,o.        ,ooooo.   \  ,'  `-') 
    )) )`. d8P"Y8.    ,8P"""""Y8.  `'  .--"' 
   (`-'   `Y'  `Y8    dP       `'     / 
    `----.(   __ `    ,' ,---.       ( 
           ),--.`.   (  ;,---.        ) 
          / \O_,' )   \  \O_,'        | 
         ;  `-- ,'       `---'        | 
         |    -'         `.           | 
        _;    ,            )          : 
     _.'|     `.:._   ,.::" `..       | 
  --'   |   .'     """         `      |`. 
        |  :;      :   :     _.       |`.`.-'--. 
        |  ' .     :   :__.,'|/       |  \ 
        `     \--.__.-'|_|_|-/        /   ) 
         \     \_   `--^"__,'        ,    | 
   -hrr- ;  `    `--^---'          ,'     | 
          \  `                    /      / 
           \   `    _ _          / 
            \           `       / 
             \           '    ,' 
              `.       ,   _,' 
                `-.___.---' 


Wat're Yeh doin in me SWAMP!!!

'''
    
# !Zach should learn what a generator is and look at the yield keyword
def newTrackingID():
    global curr_id # Don't judge me
    tmp = curr_id
    curr_id += 1
    return tmp
    
def interpolateNewPos(path):
    pathX = [num[0] for num in path if num[0] > -1] #Ignores frames where the target wasn't seen
    pathY = [num[1] for num in path if num[1] > -1]
    
    # We have not yet decided on whether or not to weight averages so leaving this in just in case
    
    #XVel = [(num - pathX[i - 1]) * (i + 1) for i,num in enumerate(pathX[1:len(pathX)])] #Weights the more the recent paths coordinates more
    #YVel = [(num - pathY[i - 1]) * (i + 1) for i,num in enumerate(pathY[1:len(pathY)])]
    
    #XAvg = sum(XVel)/sum(range(len(XVel) + 1)) if len(XVel) > 0 else 0 #Finds the average distance moved (with weighted average)
    #YAvg = (sum(YVel)/sum(range(len(YVel) + 1)) if len(YVel) > 0 else 0)
    
    # Gets an array with change of position for all targets
    XVel = [(num - pathX[i - 1]) for i,num in enumerate(pathX[1:len(pathX)])] 
    YVel = [(num - pathY[i - 1]) for i,num in enumerate(pathY[1:len(pathY)])]
    
    # Finds the average to get average velocity
    XAvg = sum(XVel)/len(XVel) if len(XVel) > 0 else 0 
    YAvg = sum(YVel)/len(YVel) if len(YVel) > 0 else 0
    
    # Adds average to the current position
    newX = pathX[-1] + XAvg 
    newY = pathY[-1] + YAvg
    
    global image # ...yeah
    
    # Plots where it predicts the next will be
    cv2.circle(image, (int(newX), int(newY)), 10, (0,0,255), thickness=-1) 
    
    return [newX, newY]

def appendToPath(path, item):
    path.append([item[0], item[1]])
    if len(path) > MAX_PATH_LEN:
        del path[0] # Deletes oldest path coordinates
        
if __name__ == "__main__":
    main()
  
# kylorenkillsdumbledore
