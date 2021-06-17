# Remove unused imports and clean modules
import argparse
import cv2
from self_func import fit_ellipse

### TAKE IMAGE INPUT AS PASSED ARGUMENT
parser = argparse.ArgumentParser(description = 'This is the OpenCV Program built for Computer Vision!')
parser.add_argument('--img', metavar = 'img', type = str, nargs = '?', default = None)
args = parser.parse_args()

### GLOBAL
cam = False
s = True
h = True
image_instance = None
l_canny = 130
scale = 10
rotate = 180
color = (255,255,255)
font = cv2.FONT_HERSHEY_SIMPLEX

### SOURCE HANDLER
if not args.img:
    print(f'No image passed, taking camera input!')
    cap = cv2.VideoCapture(0)
    cam = True
else:
    image = cv2.imread(args.img, 1)
    if image is None:
        print(f'Failed to find and load image file: {args.img}')
        exit(0)

### APPLICATION
## INIT LAGGING FRAME
if cam:
    _,prev_instance = cap.read()
    prev_instance = cv2.cvtColor(prev_instance, cv2.COLOR_BGR2GRAY)
else:
    prev_instance = image.copy()
    prev_instance = cv2.cvtColor(prev_instance, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image',prev_instance)

## INIT TRACKBAR
cv2.createTrackbar('Canny Edge Threshold','Image',l_canny,170, lambda x:x)
cv2.createTrackbar('Scale Ellipse','Image',scale,20, lambda x:x)
cv2.createTrackbar('Rotate Ellipse','Image',rotate,180, lambda x:x)

## START APPLICATION LOOP
active = True
while active:

    # Source Handle - Input
    if cam:
        _,image_instance = cap.read()
        image_instance = cv2.cvtColor(image_instance, cv2.COLOR_BGR2GRAY)
    else:
        image_instance = image.copy()
        image_instance = cv2.cvtColor(image_instance, cv2.COLOR_BGR2GRAY)
    
    # User Input
    key = cv2.waitKey(1)

    # Reset Application
    if key == ord('r'):
        l_canny = 130
        cv2.createTrackbar('Canny Edge Threshold','Image',l_canny,170, lambda x:x)
        scale = 10
        cv2.createTrackbar('Scale Ellipse','Image',scale,20, lambda x:x)
        rotate = 180
        cv2.createTrackbar('Rotate Ellipse','Image',rotate,180, lambda x:x)
        continue

    # Toggle Help Text
    if key == ord('h'):
        if h:
            h = False
        else:
            h = True

    # Exit Application
    if key == ord('q'):
        cv2.destroyAllWindows()
        active = False
        continue
    
    # Ellipse Fitting 
    if s:
        # Retrieve Edge With Canny
        l_canny = cv2.getTrackbarPos('Canny Edge Threshold','Image')
        edge = cv2.Canny(image_instance,l_canny,1.5*l_canny)
        
        # Find Frame Difference & Update Lagging Frame
        diff = cv2.subtract(image_instance, prev_instance)
        prev_instance = image_instance.copy()

        # Intersection Points b/w Edge and Difference For Fitting 
        inter = cv2.bitwise_or(edge,diff)
        _, inter = cv2.threshold(inter, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        inter = cv2.findNonZero(inter)

        # Interractive Elements
        scale = cv2.getTrackbarPos('Scale Ellipse','Image')
        rotate = cv2.getTrackbarPos('Rotate Ellipse','Image')

        # Fit The Points And Draw
        try:
            ellipse = fit_ellipse(inter, scale, rotate)
            cv2.ellipse(image_instance, ellipse, color, 1)
        except Exception as e:
            pass

    if h:
        cv2.putText(image_instance,'Ellipse Fitting To Handle',(50,50), font, 1, color,2)
        cv2.putText(image_instance,'r - Reset Application',(50,100), font, 1, color,2)
        cv2.putText(image_instance,'h - Help (toggle)',(50,150), font, 1, color,2)
        cv2.putText(image_instance,'q - Exit',(50,200), font, 1, color,2)
    
    cv2.imshow('Image',image_instance)

### CLOSE CAMERA
if cam:
    cap.release()
