# from ultralytics import YOLO

# #model = YOLO("yolov8s.pt")
# model = YOLO(r'/root/autodl-tmp/yolov11-main/runs/detect/train8/weights/best.pt')
# results = model.val(data=r"/root/autodl-tmp/yolov11-main/CCTSDB2021/CCTSDB.yaml", batch=32, imgsz=640, device=0, workers=12, split="val", amp=True)#, classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])



from ultralytics import YOLO

#model = YOLO("yolov8s.pt")
model = YOLO(r'/root/autodl-tmp/yolov11-main/runs/detect/train42/weights/best.pt')
results = model.val(data=r"/root/autodl-tmp/yolov11-main/CCTSDB2021/CCTSDB.yaml", batch=32, imgsz=640, device=0, workers=12, split="val", amp=True,iou=0.3)#, classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])