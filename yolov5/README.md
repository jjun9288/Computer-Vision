### Pistol detection using YOLO v5

- Datasets from https://public.roboflow.com/object-detection/pistols
1. Clone YOLOv5   
2. Download pistol.mp4
3. Download last.pt
4. python detect.py --source ../pistol.mp4 --weights ../last.py
5. We get the result in 'runs/detect/exp/pistol.mp4'

### Object tracking using YOLO v5

- center_point_detect.py : Getting center points from bbox
- detect_tracking.py : Tracking detected objects using camera and servo motors
