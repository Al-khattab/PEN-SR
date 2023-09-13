import cv2

# Constants
IMAGE_FILE = 'img_099.png'
PATCH_SIZE = 88
RECT_COLOR = (0, 255, 255)  # Yellow
RECT_THICKNESS = 6

def main():
    # Load the image
    image = cv2.imread(IMAGE_FILE)
    
    if image is None:
        print(f"Error: Unable to load {IMAGE_FILE}")
        return
    
    # Define the cropping coordinates
    x, y = 568, 198
    start_point = (x, y)
    end_point = (x + PATCH_SIZE, y + PATCH_SIZE)
    
    # Crop the image
    crop_img = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
    
    # Display the cropped image
    cv2.imshow("Cropped", crop_img)
    
    # Save the cropped image
    cv2.imwrite("patched_" + IMAGE_FILE, crop_img)
    
    # Draw a rectangle on the original image
    image = cv2.rectangle(image, start_point, end_point, RECT_COLOR, RECT_THICKNESS)
    
    # Display the image with the rectangle
    cv2.imshow("Image with Rectangle", image)
    
    # Save the modified image
    cv2.imwrite("bounded_" + IMAGE_FILE, image)
    
    # Wait for a key press event and release resources
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()