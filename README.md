# YOLOv4-CSP

This is a cpu only fork of [Scaled-YOLOv4-csp](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp), published as "[Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)".

## Changes

Replaced Cuda-Mish with CPU supported Mish activation function.

## Run

Download [`yolov4-csp.weights`](https://drive.google.com/file/d/1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL/view?usp=sharing) file and save it in `weights/` folder.

In a terminal run:

```bash
python inference.py --cfg models/yolov4-csp.cfg --weights weights/yolov4-csp.weights --img-src <path/to/an/image>
```

## Reference

```latex
@article{wang2020scaled,
  title={{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2011.08036},
  year={2020}
}
```
