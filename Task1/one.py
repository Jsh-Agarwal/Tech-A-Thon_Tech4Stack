# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('obj2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')


blur = cv2.GaussianBlur(gray, (11, 11), 0)
plt.imshow(blur, cmap='gray')


canny = cv2.Canny(blur, 30, 150, 3)
plt.imshow(canny, cmap='gray')


dilated = cv2.dilate(canny, np.ones((5, 5), np.uint8), iterations=1)
(cnt, hierarchy) = cv2.findContours(
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Calculate the average size of the contours
contour_areas = [cv2.contourArea(c) for c in cnt]
average_area = np.mean(contour_areas)
min_area = average_area / 2

# Filter out small contours
filtered_cnt = [c for c in cnt if cv2.contourArea(c) >= min_area]


rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

plt.imshow(rgb)


print("boxes in the image : ", len(cnt))