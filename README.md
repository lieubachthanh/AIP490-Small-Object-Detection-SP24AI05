# AIP490 - SMALL OBJECT DETECTION IN UAV IMAGES


## Install
```bash
$ git clone https://github.com/lieubachthanh/AIP490-Small-Object-Detection-SP24AI05.git
$ cd AIP490-Small-Object-Detection-SP24AI05
$ pip install -r requirements.txt
```

## Run demo
```bash
$ streamlit run demo.py
```

## Resources
* `Datasets` : [VisDrone](http://aiskyeye.com/download/object-detection-2/)
* `Weights` : [CustomYOLO](https://github.com/lieubachthanh/AIP490-Small-Object-Detection-SP24AI05/blob/main/weights/OursYOLO.pt)


## validation  
val.py runs inference on VisDrone2019-DET-val, using weights trained with OursYOLO.  

```bash
$ python val.py --weights ./weights/OursYOLO.pt --img 640 --data ./data/VisDrone.yaml --task val --batch-size 8 
```

## Train
train.py allows you to train new model from strach.
```bash
$ python train.py --img 640 --batch 8 --epochs 100 --data ./data/VisDrone.yaml --weights yolov5s.pt --cfg models/ourYolo.yaml

```

## Collaborators
- Lieu Bach Thanh
- Nguyen Huynh Lam
- Nguyen Chi Khang

## References
Thanks to their great works
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
