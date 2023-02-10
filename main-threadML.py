from time import sleep, time
from datetime import datetime, timedelta
from picamera import PiCamera
from pathlib import Path
from logzero import logger, logfile # see what went right or wrong after flight
from orbit import ISS
import exif # saving ML output data in EXIF
import cv2  # extract clouds using opencv
import numpy as np # used by opencv
import threading

# Coral ML:
from PIL import Image                                   # resize images
from pycoral.adapters import common, classify           # used by Coral Edge TPU
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
from tflite_runtime import interpreter

start_time = datetime.now()
base_folder = Path(__file__).parent.resolve()
logfile(base_folder/"events.log") #, maxBytes=50000000, backupCount=1) # the logger files shouldn't take more than ... megabytes.

def convert(angle):
    # Source: Fuction borrowed from Phase 2 guide
    """
    Convert a `skyfield` Angle to an EXIF-appropriate
    representation (positive rationals)
    e.g. 98° 34' 58.7 to "98/1,34/1,587/10"

    Return a tuple containing a boolean and the converted angle,
    with the boolean indicating if the angle is negative.
    """
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle

def capture(camera, imagePath):
    # the function found in the Phase 2 Guide, modified to return the captured image

    # log that we're capturing an image
    logger.info(f"{imagePath} - Capturing image")

    """Use `camera` to capture an `image` file with lat/long EXIF data."""
    point = ISS.coordinates()

    # Convert the latitude and longitude to EXIF-appropriate representations
    south, exif_latitude = convert(point.latitude)
    west, exif_longitude = convert(point.longitude)

    # Set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

    # Capture the image
    """
    The image is saved rather than directly pipelined into
    the Coral ML analysis in order to prvent data loss if
    something goes wrong within the analysis code.
    """
    camera.capture(imagePath)
    logger.info(f"{imagePath} - Captured image")

def isNightPhoto(imagePath):
    """
    Looks at an image at a given path, and if it is a night photo, removes it using pathlib.
    If it is a twilight photo, save this information in EXIF.
    Returns true if file was deleted.
    """
    logger.info(f"{imagePath} - Night Detection Classification Started")
    """
    Image analysis using Coral ML
    This is where the ML model bundled with this code is used.
    
    Source: Modified code from Image classification with Google Coral tutorial on projects.raspberrypi.org
    """
    script_dir = Path(__file__).parent.resolve()
    model_file = script_dir/'daynightModel.tflite'
    label_file = script_dir/'daynightLabels.txt'

    interpreter = make_interpreter(f"{model_file}")
    interpreter.allocate_tensors()

    size = common.input_size(interpreter)

    # convert to grayscale (L) to ignore changes in color, then convert back to RGB for compatibility reasons
    img = Image.open(imagePath).convert('RGB').resize(size, Image.Resampling.LANCZOS) 

    common.set_input(interpreter, img)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)

    labels = read_label_file(label_file)
    
    result = 'Day/Tw/N result'
    for c in classes:
        result = f'{labels.get(c.id, c.id)} {c.score:.5f}'

    """
        If it is a night photo, delete it.
        Else, save image, with exif data included.
    """

    if str(result).split()[0]=="night":
        Path(imagePath, missing_ok=True).unlink() # remove photo
        return True
    else:
        with open(f"{imagePath}", 'rb') as image_file:
            image = exif.Image(image_file)
        
        # Add result in 'image_description' EXIF tag
        image.make = "Octans Astro Pi (RaspberryPiHQ)" # easter egg ;)
        logger.info(f"{imagePath} - Day/Night/Twilight Result: " + str(result))
        image.image_description = str(result)
        
        # Save image
        with open(f"{imagePath}", 'wb') as saved_image:
            saved_image.write(image.get_file())
        return False

def classifyClouds(imagePath):
    """
    Classify clouds images using the Coral ML Accelerator and OpenCV

    The result is saved as EXIF inside the image at the path given
    via the `imagePath` parameter.
    """

    # log that we're starting the classification procedure
    logger.info(f"{imagePath} - CloudClassif")

    """
    HSV-based Cloud Extraction using a Color Interval

    The HSV interval denoted by following numpy arrays are used to
    extract the window and the clouds from the image, to facilitate
    cloud clasification (and NDVI in Phase 4).

    Since clouds are white and usually nothing else has a similar
    color in the image (except during twilight, when the window's
    edge is being lit), we can extract most of the clouds.

    The clouds-only image is not saved (to save space)
    and used only for cloud classsification.
    """

    img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2HSV) # convert image to HSV

    # color intervals
    cloudsUp = np.array([179, 80, 255])
    cloudsDown = np.array([0, 0, 185])

    clouds = cv2.bitwise_and(img, img, mask = cv2.inRange(img, cloudsDown, cloudsUp)) # clouds image as numpy array
    clouds = cv2.cvtColor(clouds, cv2.COLOR_HSV2RGB) # convert to RGB
    clouds = Image.fromarray(clouds) # convert numpy array image to PIL Image


    logger.info(f"{imagePath} - Cloud Classification Started")
    """
    Image analysis using Coral ML
    This is where the ML model bundled with this code is used.
    
    Source: Modified code from Image classification with Google Coral tutorial on projects.raspberrypi.org
    """
    script_dir = Path(__file__).parent.resolve()
    model_file = script_dir/'cloudModel.tflite'
    label_file = script_dir/'cloudLabels.txt'

    interpreter = make_interpreter(f"{model_file}")
    interpreter.allocate_tensors()

    size = common.input_size(interpreter)

    # convert to grayscale (L) to ignore changes in color, then convert back to RGB for compatibility reasons
    img = clouds.convert('L').convert('RGB').resize(size, Image.Resampling.LANCZOS) 

    common.set_input(interpreter, img)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)

    labels = read_label_file(label_file)
    
    result = 'Cloud type result'
    for c in classes:
        result = f'{labels.get(c.id, c.id)} {c.score:.5f}'

    """
        Save image, with exif
    """

    with open(f"{imagePath}", 'rb') as image_file:
        image = exif.Image(image_file)
    
    # Add result in 'user_comment' EXIF tag
    image.user_comment = str(result)
    logger.info(f"{imagePath} - CloudClassif Result: " + str(result))
    image.software = "0C74N5 Cloud Classification" # easter egg ;)
    
    # Save image
    with open(f"{imagePath}", 'wb') as saved_image:
        saved_image.write(image.get_file())

# initialize camera
camera = PiCamera()
camera.resolution=(2561, 1920)
photosCnt = 0

analysisCnt = 0
FIFObuffer = []
lock = threading.Lock()

def analysisThread():
    global analysisCnt
    global FIFObuffer
    while True:
        try:
            if len(FIFObuffer) != 0:
                
                filename = ""
                with lock:
                    analysisCnt += 1
                    filename = FIFObuffer[0]
                if not isNightPhoto(filename): # if it is not a night photo (if it is, it is automatically deleted)
                    logger.info(f"{filename} - Night not detected")
                    classifyClouds(filename)
                else:
                    logger.info(f"{filename} - Night detected, photo deleted.")
                with lock:
                    FIFObuffer.remove(filename)
        except Exception as e:
            # Log exception, save debug info (number of photos taken, ).
            logger.debug(f"mlanalysisd T:{time()} deltaT:{time()-start_time.second} P:{photosCnt: .0f}")
            logger.error(e)
            pass

x = threading.Thread(target=analysisThread, daemon=True)
x.start()

while (datetime.now() < start_time + timedelta(minutes=178)) and photosCnt <= 1500:
    """
        Take photos every 7.2 seconds, analyse photos using Coral (and delete them if it is a night photo), log exceptions if there are any.
    """
    try:
        photoTime = round(time(), 3) # save starting photo time to sleep for only as much time as needed (ML analysis might be slower than expected)
        photosCnt = photosCnt+1

        filename = f"{base_folder}/OCTANS_{photosCnt}.jpg"
        capture(camera, filename) # capture photo
        with lock:
            FIFObuffer.append(filename) # add to ml analysis thread

        analysisTime = round(time() - photoTime, 2) # time taken by ML analysis
        if 7.2-analysisTime > 0: # if time didn't pass
            sleep(7.2-analysisTime)

    except Exception as e:
        # Log exception, save debug info (number of photos taken, ).
        logger.debug(f"T:{time()} deltaT:{time()-start_time.second} P:{photosCnt: .0f}")
        logger.error(e)
        pass

# closing gracefully
with lock:
    logger.info(f"Closing program. {photosCnt} photos taken, {analysisCnt} ML-analysed.")
camera.close()
