# Automatic Detection and Classification System of Domestic Waste via a Cascaded YOLOv8 and ResNet-101 CNN

A two-stage computer-vision pipeline for domestic waste recognition. **YOLOv8** detects waste objects, each detected region is cropped, and **ResNet-101** classifies the crop. A detection is kept only when the ResNet-101 class **agrees** with the YOLOv8 class (agreement-based verification).

**Authors:** Mrityunjay Yadav, Gaurav Yadav
**Affiliation:** Department of Information Technology, Netaji Subhas University of Technology (NSUT), New Delhi, India

## Pipeline

```
Input image -> YOLOv8 detection -> crop extraction -> ResNet-101 classification -> class-agreement filter -> final detections
```

- Classes (5): Glass, Metal, Paper, Plastic, Waste
- ResNet-101 input: 224x224, ImageNet normalization (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225])
- The ResNet-101 stage only verifies YOLOv8 boxes; it cannot recover objects YOLOv8 missed, and it does not relabel detections.

## Dataset

Trash-Detection-14 from Roboflow Universe (YOLO-format annotations). The dataset is **not** included in this repository because of size and licensing. Download it from: `<ADD EXACT ROBOFLOW DATASET URL AND VERSION HERE>`

- Detection test set: 110 images (175 annotated objects)
- ResNet-101 patches: 13,042 train / 1,414 validation / 175 test

## Results (110 test images)

| Approach | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Time (s/image) | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| YOLOv8 | 0.6571 | 0.5257 | 0.4678 | 0.4338 | 0.0107 | 92 | 48 | 83 |
| YOLOv8 + ResNet-101 | 0.8980 | 0.5029 | 0.4533 | 0.4193 | 0.0304 | 88 | 10 | 87 |

Per-class AP@0.5 (YOLOv8 / cascade): Glass 0.4651 / 0.4500, Metal 0.6270 / 0.6376, Paper 0.6019 / 0.5556, Plastic 0.6452 / 0.6235, Waste 0.0000 / 0.0000.

**Takeaway:** the cascade raises precision (0.6571 to 0.8980) and cuts false positives (48 to 10), but recall and mAP drop slightly and inference time is about 2.84x higher. It is useful when avoiding false detections matters more than catching every object.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── paper/
│   ├── main.tex        # IEEE conference paper (LaTeX source)
│   └── main.pdf        # compiled paper
├── notebooks/          # training / inference / evaluation notebooks
└── weights/            # best.pt (YOLOv8), best_resnet101.pth (ResNet-101)
```

## Setup

```bash
git clone https://github.com/<your-username>/Garbage-Detection-YOLOv8-ResNet101.git
cd Garbage-Detection-YOLOv8-ResNet101
pip install -r requirements.txt
```

Environment used for the experiments: Python 3.13.15, PyTorch 2.11.0+cu130, Ultralytics 8.4.175, NVIDIA Tesla T4 GPU.

## Training details

| Item | Value |
|---|---|
| YOLOv8 variant / image size | `<fill in>` |
| Epochs / batch size | `<fill in>` |
| Optimizer / learning rate | `<fill in>` |

## Compiling the paper

Upload `paper/main.tex` to Overleaf (pdfLaTeX), or run `pdflatex main.tex` twice inside `paper/`.

## Citation

```bibtex
@misc{yadav2026waste,
  title  = {Automatic Detection and Classification System of Domestic Waste via a Cascaded YOLOv8 and ResNet-101 Convolutional Neural Network},
  author = {Yadav, Mrityunjay and Yadav, Gaurav},
  year   = {2026}
}
```

## License

`<Choose a license, e.g. MIT, and add a LICENSE file>`
