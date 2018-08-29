# About
A vision resolution algorithm written for the 2016-17 FRC (First Robotics Competition) year. The algorithm is written in python and uses OpenCV to resolve and track vision targets (basically just retroreflective tape illuminated by a green LED).

# Hardware
- Nvidia Jetson - A microprocessor by Nvidia specifically for vision processing purposes such as this
- RoboRIO - A microprocessor built by NI and provided by the FRC

# How it works
Each vision target is resolved through a process of image filtering, color filtering, masking and finally resolution of bounding boxes from the masks. Targets that would be considered too "small" are considered artifacts and removed from the list. Each target is given a unique "Tracking ID" and is tracked from frame to frame based on position movements and size changes. This tracking ID is then sent along with the bounding box coordinates over NetworkTables, a proprietary network communication system that runs on the RoboRio (a microprocessor provided by FIRST, the company that runs FRC).
