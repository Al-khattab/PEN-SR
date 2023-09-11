import cv2

image = cv2.imread('img_099.png')
print(image.shape)
x = 568
y = 198
start_point = (x,y )
  
# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
size = 88
end_point = (x + size ,y+ size )
  
# Blue color in BGR
color = (0, 255, 255)
# Line thickness of 2 px
thickness = 6
crop_img = image[y:y+size, x:x+size]
cv2.imshow("cropped", crop_img)
cv2.imwrite("patched_img_099.png",crop_img)
# Using cv2.rectangle() method
# Draw a rectangle with blue line borders of thickness of 2 px
image = cv2.rectangle(image, start_point, end_point, color, thickness)
  

cv2.imwrite("bounded_img_099.png",image)

cv2.imshow("hi", image)
k = cv2.waitKey(0) # 0==wait forever
