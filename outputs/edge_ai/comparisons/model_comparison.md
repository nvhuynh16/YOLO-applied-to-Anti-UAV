# YOLOv8n Edge AI Model Comparison

## Performance Comparison Table

| Model Variant        |   Size (MB) |   Parameters (M) |   FLOPs (G) |   CPU FPS |   GPU FPS |   mAP50 (%) |   mAP50-95 (%) | Use Case                  | Target Hardware    |
|:---------------------|------------:|-----------------:|------------:|----------:|----------:|------------:|---------------:|:--------------------------|:-------------------|
| YOLOv8n              |        5.93 |                3 |         8.1 |      11.5 |        65 |        99.5 |           83.2 | Development & Testing     | Any (GPU/CPU)      |
| (PyTorch FP32)       |             |                  |             |           |           |             |                |                           |                    |
| YOLOv8n              |       11.7  |                3 |         8.1 |      15   |        80 |        99.5 |           83.2 | Cross-Platform Deployment | Any (ONNX Runtime) |
| (ONNX)               |             |                  |             |           |           |             |                |                           |                    |
| YOLOv8n              |        3    |                3 |         8.1 |      12   |        95 |        99.5 |           83.2 | GPU Edge Devices          | NVIDIA Jetson, RTX |
| (FP16)               |             |                  |             |           |           |             |                |                           |                    |
| YOLOv8n              |        1.5  |                3 |         8.1 |      18   |        90 |        98.5 |           81   | CPU Edge Devices          | CPU (ARM, x86)     |
| (INT8)               |             |                  |             |           |           |             |                |                           |                    |
| YOLOv8n-416          |        5.93 |                3 |         3.5 |      25   |       140 |        98   |           79.5 | Real-Time Applications    | Any (GPU/CPU)      |
| (Reduced Resolution) |             |                  |             |           |           |             |                |                           |                    |

## Deployment Recommendations

### YOLOv8n
(PyTorch FP32)
- **Size**: 5.93 MB
- **Performance**: 11.5 FPS (CPU), 65.0 FPS (GPU)
- **Accuracy**: 83.2% mAP50-95
- **Target Hardware**: Any (GPU/CPU)
- **Best Use Case**: Development & Testing

### YOLOv8n
(ONNX)
- **Size**: 11.7 MB
- **Performance**: 15.0 FPS (CPU), 80.0 FPS (GPU)
- **Accuracy**: 83.2% mAP50-95
- **Target Hardware**: Any (ONNX Runtime)
- **Best Use Case**: Cross-Platform Deployment

### YOLOv8n
(FP16)
- **Size**: 3.0 MB
- **Performance**: 12.0 FPS (CPU), 95.0 FPS (GPU)
- **Accuracy**: 83.2% mAP50-95
- **Target Hardware**: NVIDIA Jetson, RTX
- **Best Use Case**: GPU Edge Devices

### YOLOv8n
(INT8)
- **Size**: 1.5 MB
- **Performance**: 18.0 FPS (CPU), 90.0 FPS (GPU)
- **Accuracy**: 81.0% mAP50-95
- **Target Hardware**: CPU (ARM, x86)
- **Best Use Case**: CPU Edge Devices

### YOLOv8n-416
(Reduced Resolution)
- **Size**: 5.93 MB
- **Performance**: 25.0 FPS (CPU), 140.0 FPS (GPU)
- **Accuracy**: 79.5% mAP50-95
- **Target Hardware**: Any (GPU/CPU)
- **Best Use Case**: Real-Time Applications

