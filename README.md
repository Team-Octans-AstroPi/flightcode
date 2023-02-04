# Flightcode
Code that runs during AstroPi Phase 2.

`main.py` is the normal code.
`main-threadML.py` is a modified version of `main.py` that uses a separate thread to do the Coral ML classifications. This eliminates the impact on the rate at which photos are taken.

Photos are saved first (with their coordinates in EXIF), before being analysed by Coral. This is to ensure data is saved even if an error occurs regarding the ML code.
Then, they are classified using the Coral USB Accelerator. If they are made during the night, they are deleted, else, classification results are saved into exif.