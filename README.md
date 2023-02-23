# Flightcode
Code that runs during AstroPi Phase 2.

`main.py` is the normal code.
`main-threadML.py` is a modified version of `main.py` that uses a separate thread to do the Coral ML classifications. This eliminates the impact on the rate at which photos are taken.

Photos are saved first (with their coordinates in EXIF), before being analysed by Coral. This is to ensure data is saved even if an error occurs in the ML code.
Then, they are classified using the Coral USB Accelerator. If they are made during the night, they are deleted, else, classification results are saved into the photos' EXIF.

## Photo EXIF Tags:
Camera brand: `make`

Camera model: `model`

Software used: `software`

GPS: `gps_latitude`, `gps_longitude`

Day/Night/Twilight: `image_description`

Clouds: `user_comment`

## Before-flight test results
The `main.py` program was tested by placing the Raspberry Pi HQ Camera in front of a screen, playing a video with IR Photos from the Raspberry Pi Foundation's Flickr.
- 1484 photos at 1440x1080 resolution
- less than 1 GB of memory occupied, including the source code, log file, and photos
- program stopped after 2h 58 min 6s
- no errords were found in the log
