import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import self_func

### Take filename as an passed argument during initialization
parser = argparse.ArgumentParser(description = 'This is the OpenCV Program built for Computer Vision!')
parser.add_argument('--img', metavar = 'img', type = str, nargs = '?', default = None)
args = parser.parse_args()

### Read from a file or capture from camera
if not args.img:
    print(f'No image passed, taking photo!')
    cap = cv2.VideoCapture(0)  ### <- change this before submit
    _,image = cap.read()
    cap.release()
else:
    image = cv2.imread(args.img, 1)

### Print Shape
print(f'Shape of image is {image.shape}')

### Trackbar for Smoothing using openCV 
def sTrackBar(n):
    global image_instance
    kernel = np.ones((n,n),np.float32)/(n*n)
    try:
        image_instance = cv2.filter2D(image_instance_s,-1,kernel)
        cv2.imshow('Image',image_instance)
    except Exception:
        pass

### Trackbar for Smoothing using self written code
def smoothingBar(n):
    global image_instance
    image_instance = self_func.smoothing(n, image_instance_S)
    cv2.imshow('Image', image_instance)

### Trackbar for rotating images
def rotateBar(degree=0):
    global image_instance
    rotation_matrix = cv2.getRotationMatrix2D((r_width / 2, r_height / 2), degree, 1)
    image_instance = cv2.warpAffine(image_instance_r, rotation_matrix, (r_width, r_height))
    cv2.imshow('Image', image_instance)

### Trackbar for changing N in gradients
def gradientBar(gd=5):
    plt.close('all')
    global image_instance
    r, c = image_instance.shape
    image_instance_p = image_instance[::gd,::gd]  # sampling

    fig,ax=plt.subplots(1,1)
    the_image = ax.imshow(
                    image_instance,
                    zorder=0,alpha=1.0,
                    cmap="Greys_r",
                    origin="upper",
                    interpolation="hermite",
                )
    plt.colorbar(the_image)            
    Y, X = np.mgrid[0:r:gd, 0:c:gd]
    dY, dX = np.gradient(image_instance_p)
    ax.quiver(X, Y, dX, dY, color='gray')
    plt.show()

### MAIN LOOP
image_instance = image.copy()
active = True
while active:
    # Take input
    cv2.imshow('Image',image_instance)
    key = cv2.waitKey(0)

    # save image
    if key == ord('w'):
        cv2.imwrite('./out.jpg', image_instance)

    image_instance = image.copy()

    # reset image
    if key == ord('i'):
        cv2.destroyAllWindows()
        image_instance = image.copy()
        continue
    
    # grayscale using opencv
    elif key == ord('g'):
        try:
            image_instance = cv2.cvtColor(image_instance, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
    
    # grayscale by self
    elif key == ord('G'):
        G = np.array([[[ 0.07, 0.72,  0.21]]])
        image_instance = np.sum(image_instance*G, axis=2)
        image_instance = (image_instance - np.min(image_instance)) / (np.max(image_instance) - np.min(image_instance))

    # cycle through color channels
    elif key == ord('c'):
        for c_index in range(3):
            channel = np.zeros(shape=image_instance.shape, dtype=np.uint8)
            channel[:,:,c_index] = image_instance[:,:,c_index]
            cv2.imshow(f'Channel {c_index} - RGB', channel)
            cv2.waitKey(0)
            cv2.destroyWindow(f'Channel {c_index} - RGB')

    # convert to grayscale and smooth it using openCV function with trackbar
    elif key == ord('s'):
        try:
            image_instance = cv2.cvtColor(image_instance, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
        image_instance_s = image_instance.copy()
        cv2.createTrackbar('Smoothing','Image',0,50,sTrackBar)
        sTrackBar(0)

    # convert to grayscale and smooth it with self written function
    elif key == ord('S'):
        try:
            image_instance = cv2.cvtColor(image_instance, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
        image_instance_S = image_instance.copy()
        cv2.createTrackbar('Smooth','Image',1,20,smoothingBar)
        smoothingBar(1)
    
    # downsample image without smoothing
    elif key == ord('d'):
        image_instance = image_instance[::2,::2]

    # downsample image with smoothing
    elif key == ord('D'):
        image_instance = cv2.resize(image_instance, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # convert to grayscale and perform convolution with an x derivative filter. Normalize to get range[0,255]. 
    elif key == ord('x'):
        try:
            image_instance = cv2.cvtColor(image_instance, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
        image_instance = cv2.Sobel(image_instance, cv2.CV_64F, 1, 0, ksize=3)
        image_instance = cv2.convertScaleAbs(image_instance)
        image_instance = np.uint8(image_instance)
    
    # convert to grayscale and perform convolution with an y derivative filter. Normalize to get range[0,255].
    elif key == ord('y'):
        try:
            image_instance = cv2.cvtColor(image_instance, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
        image_instance = cv2.Sobel(image_instance, cv2.CV_64F, 0, 1, ksize=3)
        image_instance = cv2.convertScaleAbs(image_instance)
        image_instance = np.uint8(image_instance)

    # magnitude of gradient normalized to range [0,255]
    elif key == ord('m'):
        try:
            image_instance = cv2.cvtColor(image_instance, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
        dX = cv2.Sobel(image_instance, cv2.CV_64F, 1, 0, ksize=3)
        dX = cv2.convertScaleAbs(dX)
        dX = np.uint8(dX)
        dY = cv2.Sobel(image_instance, cv2.CV_64F, 0, 1, ksize=3)
        dY = cv2.convertScaleAbs(dY)
        dY = np.uint8(dY)
        image_instance = cv2.addWeighted(dX, 0.5, dY, 0.5, 0)
    
    # Plot gradient vertors with variable N using trackbar 
    elif key == ord('p'):
        try:
            image_instance = cv2.cvtColor(image_instance, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
        cv2.createTrackbar('Gradient','Image',5,20,gradientBar)
        gradientBar(5)
    
    # rotate image using trackbar
    elif key == ord('r'):
        try:
            image_instance = cv2.cvtColor(image_instance, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
        image_instance_r = image_instance.copy()
        r_height, r_width = image_instance.shape[:2]
        cv2.createTrackbar('Degree','Image',0,360,rotateBar)
        rotateBar(0)

    # display all commands
    elif key == ord('h'):
        print(f'Help')
    
    # Exit Case
    elif key == 27:
        print(f'Exiting')
        cv2.destroyAllWindows()
        active = False
