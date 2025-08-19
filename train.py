
# from ultralytics import YOLO
# import warnings
# warnings.filterwarnings('ignore')
# if __name__ == '__main__':
#    model = YOLO(r"\ultralytics\cfg\models\11\yolo11.yaml")

#    results = model.train(data="/CCTSDB2021/CCTSDB2021.yaml", imgsz=640,epochs=200, batch=32, device=0, optimizer="SGD", workers=12)
#    metrics=model.val()




from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
   model = YOLO(r"ultralytics/cfg/models/11/yolo11-NewConvBlock.yaml")

   results = model.train(data="/GTSDB/GTSDB.yaml", imgsz=640,epochs=200, batch=32, device=0, optimizer="SGD", workers=12)
   metrics=model.val()
