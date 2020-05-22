import cv2
import numpy as np

####### GRAYSCALING IMAGES #######

image = cv2.imread('./imgs/dravid.jpg')

height, width = image.shape[:2]

cv2.imshow('Original', image)
cv2.waitKey()


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('./imgs/gray_scale_dravid.jpg', gray_image)


####### HISTOGRAM REPRESENTATION OF IMAGES #######
import matplotlib.pyplot as plt

color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histogram = cv2.calcHist([image], [i], mask = None, histSize = [256], ranges = [0,256])
    plt.plot(histogram, color = col)
    plt.xlim([0,256])
plt.show()

plt.savefig('./plots/color_dist.png')


####### IMAGE TRANSLATIONS #######

# We apply an affine transformation that shifts the position of an image

quarter_height, quarter_width = height/4, width/4

#       / 1 0 Tx /
#  T =  / 0 1 Ty /

# T is the translation matrix
T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

# We use warpAffine to transform the image using the matrix T

translated_image = cv2.warpAffine(image, T, (width, height))
cv2.imshow('Translation', translated_image)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('./imgs/translated_dravid.jpg', translated_image)


####### IMAGE ROTATIONS #######

# Rotation Matrix = |cos (theta) -sin(theta)|
#                   |sin (theta) cos(theta)|

# We are taking theta = 45degrees here
# Dividing by two to rotate the image around its centre
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)

print(rotation_matrix)

rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

cv2.imshow('Rotation', rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('./imgs/rotated_dravid.jpg', rotated_image)


####### SCALING, RESIZING AND INTERPOLATIONS #######

# By default, cv2.resize uses Linear Interpolation
# Resizing the image to 3/4th of its original size
scaled_image = cv2.resize(image, None, fx=0.75, fy=0.75)

cv2.imshow('Scaling - Linear Interpolation', scaled_image)
cv2.waitKey()

cv2.imwrite('./imgs/interpolations/linear.jpg', scaled_image)

# Now, we will be using the Inter-Cubic interpolation
# Doubling up the original image size
scaled_image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

cv2.imshow('Scaling - INTER_CUBIC', scaled_image)
cv2.waitKey()

cv2.imwrite('./imgs/interpolations/inter_cubic.jpg', scaled_image)

# Next, we try out the Inter-Area interpolation
# Resizing the image by setting exact dimensions
scaled_image = cv2.resize(image, (900,400), interpolation=cv2.INTER_AREA)

cv2.imshow('Scaling - INTER_AREA', scaled_image)
cv2.waitKey()

cv2.imwrite('./imgs/interpolations/inter_area.jpg', scaled_image)

cv2.destroyAllWindows()


####### IMAGE PYRAMIDS #######

# Useful when scaling images in object detection

smaller_image = cv2.pyrDown(image)
larger_image = cv2.pyrUp(image)

cv2.imshow('Smaller', smaller_image)
cv2.imshow('Larger', larger_image)
cv2.waitKey()

cv2.destroyAllWindows()

####### IMAGE CROPPING #######

# Starting pixel co-ordinates (top left of the cropping rectangle)
start_row, start_col = int(height*.25), int(width*0.25)

# Ending pixel co-ordinates (bottom right of the cropping rectangle)
end_row, end_col = int(height*.75), int(width*.75)

# Simply use indexing to crop out the rectangle 
cropped_image = image[start_row:end_row, start_col:end_col]

cv2.imshow('Cropped', cropped_image)
cv2.waitKey()
cv2.destroyAllWindows()


####### ARITHMETIC OPERATIONS #######

# Creating a matrix of ones, then multiply by a scalar of 75
# This gives a matrix with same dimensions of our image with all values being 75
M = np.ones(image.shape, dtype="uint8")*75

# We use the cv2.add function to add the matrix M to our image
# Notice the increase in brightness
added_image = cv2.add(image, M)
cv2.imshow("Added", added_image)
cv2.imwrite('./imgs/added_dravid.jpg', added_image)

# Likewise we subtract the matrix M
# The brightness will decrease now
subtracted_image = cv2.subtract(image, M)
cv2.imshow('Subtracted', subtracted_image)
cv2.imwrite('./imgs/subtracted_dravid.jpg', subtracted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

####### IMAGE BLURRING #######

# Creating our 3x3 kernel
kernel_3 =  np.ones((3,3), np.float32)/9

# We use the cv2.filter2d to convolve the kernel with an image
blurred_image = cv2.filter2D(image, -1, kernel_3)
cv2.imshow('3x3 Kernel Blurring', blurred_image)

cv2.imwrite('./imgs/blurred/3x3_blurred_dravid.jpg', blurred_image)

# Creating our 5x5 kernel
kernel_5 = np.ones((5,5), np.float32)/25

blurred_image = cv2.filter2D(image, -1, kernel_5)
cv2.imshow('5x5 Kernel Blurring', blurred_image)

cv2.imwrite('./imgs/blurred/5x5_blurred_dravid.jpg', blurred_image)

# Creating our 7x7 kernel
kernel_7 = np.ones((7,7), np.float32)/49

blurred_image = cv2.filter2D(image, -1, kernel_7)
cv2.imshow('7x7 Kernel Blurring', blurred_image)

cv2.imwrite('./imgs/burred/7x7_blurred_dravid.jpg', blurred_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

### Other commonly used blurring methods

""" 
    Averaging done by convolving the image with a normalised box filter.
    This takes the pixels under the box and replaces the central element. Box size
    needs to add and be positive 
"""

blur = cv2.blur(image, (3,3))
cv2.imshow('Averaging', image)

cv2.imwrite('./imgs/blurred/avg_blur_dravid.jpg', blur)

# Using Gaussian kernel
gaussian = cv2.GaussianBlur(image, (7,7), 0)
cv2.imshow('Gaussian', gaussian)

cv2.imwrite('./imgs/blurred/gaussian_blur_dravid.jpg', gaussian)


""" 
MedianBlur takes the median of all the pixels under kernel area  and
central element is replaced with this median value 

"""

median = cv2.medianBlur(image, 5)
cv2.imshow('Median', median)

cv2.imwrite('./imgs/blurred/median_blur_dravid.jpg', median)


# Bilateral is very effective in noise removal while keeping edges sharp
bilateral = cv2.bilateralFilter(image, 9, 175, 175) # setting high sigma values for "cartoonish" effects :)
cv2.imshow('Bilateral', bilateral)

cv2.imwrite('./imgs/blurred/bilateral_blur_dravid.jpg', bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()


####### IMAGE SHARPENING #######

# Creating a sharpening kernel
# We don't need to normalize it as the sum is equal to 1 itself

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

image_sharpened = cv2.filter2D(image, -1, kernel)

cv2.imshow('Sharpening', image_sharpened)
cv2.imwrite('./imgs/sharpened_dravid.jpg', image_sharpened)

cv2.waitKey()
cv2.destroyAllWindows()


####### THRESHOLDING, BINARIZATION & ADAPTIVE THRESHOLDING #######

# We have to use the grayscale image for further thresholding and binarization

# Values below 127 goes to 0 (black) and everything else 255(white)
ret, image_th1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', image_th1)
cv2.imwrite('./imgs/thresholds/dravid_th1.jpg', image_th1)

# Values above 127 goes to 0 (black) and everything else 255(white) {Inverse of THRESH_BINARY}
ret, image_th2 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Threshold Binary Inverse', image_th2)
cv2.imwrite('./imgs/thresholds/dravid_th2.jpg', image_th2)

# Values above 127 are truncated at 127 
ret, image_th3 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow('Threshold Truncated', image_th3)
cv2.imwrite('./imgs/thresholds/dravid_th3.jpg', image_th3)

# Values below 127 go to 0, above are unchanged
ret, image_th4 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow('Threshold To Zero', image_th4)
cv2.imwrite('./imgs/thresholds/dravid_th4.jpg', image_th4)

# Values above 127 go to 0, below are unchanged {Inverse of THRESH_TOZERO}
ret, image_th5 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('Threshold To Zero Inverse', image_th5)
cv2.imwrite('./imgs/thresholds/dravid_th5.jpg', image_th5)

cv2.waitKey()
cv2.destroyAllWindows()


# A simpler and better way to binarize is by using Adaptive Thresholding

# Blurring the image to remove noise 

gray_image_blurred = cv2.GaussianBlur(gray_image, (3,3), 0)

# Using Adaptive threshold 

## Based on mean of neighbouring pixels
image_th_mean = cv2.adaptiveThreshold(gray_image_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY, 3, 5)
cv2.imshow('Adaptive Mean Thresholding', image_th_mean)
cv2.imwrite('./imgs/thresholds/dravid_ad_th_mean.jpg', image_th_mean)

# Otsu's Thresholding
ret, image_th_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu's Thresholding",image_th_otsu)
cv2.imwrite('./imgs/thresholds/dravid_otsu_thresh.jpg', image_th_otsu)

# Otsu's Thresholding after blurring
ret, image_blur_th_otsu = cv2.threshold(gray_image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Gaussian Otsu's Thresholding",image_th_otsu)
cv2.imwrite('./imgs/thresholds/dravid_blurred_otsu_thresh.jpg', image_blur_th_otsu)

cv2.waitKey()
cv2.destroyAllWindows()


####### DILATION, EROSION, OPENING AND CLOSING #######

# Defining our kernel size
kernel = np.ones((5,5), np.uint8)

# We apply erosion (Removing pixels from boundary of object)
image_eroded = cv2.erode(image, kernel, iterations=1)
cv2.imshow('Erosion', image_eroded)
cv2.imwrite('./imgs/dravid_eroded.jpg', image_eroded)

# We apply dilation (Adding pixels to boundary of object)
image_dilated = cv2.dilate(image, kernel, iterations=1)
cv2.imshow('Dilation', image_dilated)
cv2.imwrite('./imgs/dravid_dilated.jpg', image_dilated)

# Opening (Good for removing noise)
image_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', image_opened)
cv2.imwrite('./imgs/dravid_opened.jpg', image_opened)

# Closing (Good for removing noise)
image_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', image_closed)
cv2.imwrite('./imgs/dravid_closed.jpg', image_closed)


cv2.waitKey()
cv2.destroyAllWindows()

####### EDGE DETECTION AND IMAGE GRADIENTS #######

image = gray_image

# Extracting Sobel Edges (emphasizes vertical or horizontal edges)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

cv2.imshow('Sobel X', sobel_x)
cv2.imwrite('./imgs/edges/dravid_sobel_x.jpg', sobel_x)

cv2.imshow('Sobel Y', sobel_y)
cv2.imwrite('./imgs/edges/dravid_sobel_y.jpg', sobel_y)

cv2.waitKey()

sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow('Sobel OR', sobel_OR)
cv2.imwrite('./imgs/edges/dravid_sobel_OR.jpg', sobel_OR)

cv2.waitKey()

# Extracting Laplacian features (gets all orientations)
laplacian = cv2.Laplacian(image, cv2.CV_64F)
cv2.imshow('Laplacian', laplacian)
cv2.imwrite('./imgs/edges/dravid_laplacian.jpg', laplacian)

cv2.waitKey()

# Canny Edge Detection using gradient values as thresholds
canny = cv2.Canny(image, 85, 170)
cv2.imshow('Canny', canny)
cv2.imwrite('./imgs/edges/dravid_canny.jpg', canny)

cv2.waitKey()
cv2.destroyAllWindows()


####### CONTOURS #######

#Finding contours

blank_image = np.zeros((height, width, 3))

canny = cv2.Canny(gray_image, 70, 255)

contours, heirarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('Image', canny)

cv2.drawContours(blank_image, contours, -1, (0,255,0), 2)

cv2.imshow('Contours', blank_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


## Approximating Contours
blank_image = np.zeros((height,width,3))

cv2.waitKey(0)

r_areas = [cv2.contourArea(c) for c in contours]
max_area = np.max(r_areas)

for c in contours:
    if((cv2.contourArea(c) > max_area * 0.01) and (cv2.contourArea(c) < max_area) ):
        cv2.drawContours(blank_image,[c], -1, (0,255,0))

cv2.imshow('Contours', blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
