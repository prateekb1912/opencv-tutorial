## Importing the necessary packages
from perspective.transform import four_pt_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import imutils
import cv2

## Constructing the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())



#### Step 1: Edge Detection ####
## Loading the image and computing the ratio of old and new heights
## and then resizing the image
img = cv2.imread(args['image'])
ratio = img.shape[0]/500
img2 = imutils.resize(img, height=500)

## Converting the image to grayscale to find the edges
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

## Displaying the original image and the edge detected one.
print("STEP 1: Edge Detection")
cv2.imshow("Original", imutils.resize(img, height=500))
cv2.imshow("Edges", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()



#### Step 2: Finding Contours ####
## Finding the contours in the edge detected image, and keeping only the largest ones
conts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
conts = imutils.grab_contours(conts)
conts = sorted(conts, key=cv2.contourArea, reverse=True)[:5]
## Looping over the Contours
for c in conts:
    ## Approximating the Contours
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    ## If the approximated contour has 4 points, then we can assume to have found the document
    if(len(approx) == 4):
        screenCnt = approx
        break
## Display the contours of the document
print("STEP 2: Finding Contours")
cv2.drawContours(img2, [screenCnt], -1, (255,0,0), 2)
cv2.imshow("Outlines", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()



#### Step 3: Applying Perspective Transform and Threshold
## Applyng the 4-point transformfor a top-down view
warped = four_pt_transform(img, screenCnt.reshape(4,2)*ratio)

## Converting the warped image to grayscale, and then thresholding to give a 'B&W' effect.
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

## Displaying the original and scanned images
print("STEP 3: Applying Perspective Transform and Threshold")
cv2.imshow("Original", imutils.resize(img, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height= 650))
cv2.imwrite('scanned.jpg', warped)
cv2.waitKey(0)
