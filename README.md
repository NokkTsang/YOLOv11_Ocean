# YOLOv11_Ocean - Marine Life Detection Benchmark

This repository provides a comprehensive benchmark of multiple object detection models for marine life detection tasks, delivering performance evaluations and trained models.

## Project Overview

This project evaluates the performance of various state-of-the-art object detection models on marine life detection tasks.

### Model List
- **YOLOv8n** - YOLOv8 Nano
- **YOLOv8s** - YOLOv8 Small
- **YOLOv10** - YOLOv10
- ~~**YOLOv11n** - YOLOv11 Nano~~
- **RT-DETR** - Real-Time Detection Transformer
- **Faster R-CNN** - Classic two-stage detector

## Evaluation Metrics

### 1. Detection Performance
- **mAP50** - Mean Average Precision at IoU=0.5
- **mAP50-95** - Mean Average Precision at IoU=0.5:0.95
- **mAPS** - Small Objects mAP
- **Recall** - Detection Recall Rate

### 2. Efficiency & Complexity
- **FPS** - Frames Per Second (Inference Speed)
- **Params** - Model Parameters (M)
- **GFLOPs** - Computational Complexity

## Experimental Results

### Detection Performance Comparison

| Model | mAP50 | mAP50-95 | mAPS | Recall | Config |
|-------|-------|----------|------|--------|--------|
| YOLOv8n | - | - | - | - | [config](configs/yolov8n.yaml) |
| YOLOv8s | - | - | - | - | [config](configs/yolov8s.yaml) |
| YOLOv10 | - | - | - | - | [config](configs/yolov10.yaml) |
| RT-DETR | - | - | - | - | [config](configs/rtdetr.yaml) |
| Faster R-CNN | - | - | - | - | [config](configs/faster_rcnn.yaml) |

### Efficiency & Complexity Comparison

| Model | FPS | Params (M) | GFLOPs | Input Size |
|-------|-----|------------|--------|------------|
| YOLOv8n | - | - | - | - |
| YOLOv8s | - | - | - | - |
| YOLOv10 | - | - | - | - |
| RT-DETR | - | - | - | - |
| Faster R-CNN | - | - | - | - |

*Note: FPS tested on [GPU Model] + [CUDA Version]*

## Project Structure

```
YOLOv11_Ocean/
│
├── README.md                 # Project documentation
│
├── data/                     # Dataset directory
│   ├── train/               # Training set
│   ├── val/                 # Validation set
│   ├── test/                # Test set
│   └── data.yaml            # Dataset configuration
│
├── configs/                  # Model configuration files
│   ├── yolov8n.yaml
│   ├── yolov8s.yaml
│   ├── yolov10.yaml
│   ├── rtdetr.yaml
│   └── faster_rcnn.yaml
│
├── models/                   # Trained model weights
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   ├── yolov10.pt
│   ├── rtdetr.pt
│   └── faster_rcnn.pth
│
├── scripts/
│   ├── train_yolov8n.py
│   ├── train_yolov8s.py
│   ├── train_yolov10.py
│   ├── train_rtdetr.py
│   ├── train_faster_rcnn.py
│   └── evaluate.py
│
└── results/
```

## Quick Start

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. Place the dataset in the `data/` directory
2. Ensure data format follows YOLO or COCO format
3. Update the `data/data.yaml` configuration file


## Requirements

```
Python >= 3.8
PyTorch >= 2.0.0
ultralytics >= 8.0.0
mmdetection >= 3.0.0
numpy
opencv-python
matplotlib
tensorboard
...
```

## Experimental Setup

- **Input Size**: -
- **Batch Size**: -
- **Training Epochs**: -
- **Optimizer**: AdamW (YOLO series) / SGD (Faster R-CNN)
- **Learning Rate**: -
- **Data Augmentation**: Mosaic, MixUp, HSV augmentation
- **Hardware**: [GPU Model to be specified]

## Citation

If this project helps your research, please cite:

```bibtex
@misc{yolov11_ocean2025,
  title={YOLOv11 Ocean: xxx},
  author={Nokk Tsang},
  year={2025},
  howpublished={\url{https://github.com/NokkTsang/YOLOv11_Ocean}}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.
