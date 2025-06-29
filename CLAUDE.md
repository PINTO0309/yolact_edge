# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLACT Edge is a real-time instance segmentation framework optimized for edge devices (NVIDIA Jetson). It's a PyTorch-based deep learning project implementing the paper "YolactEdge: Real-time Instance Segmentation on the Edge" (ICRA 2021).

**Current Branch**: `samacoco_person` - Configured for person-only detection using custom SAMA-COCO dataset.

## Key Commands

### Building the Project
```bash
# Install Cython extensions (required after cloning)
pip install cython
python setup.py build_ext --inplace
```

### Training
```bash
# Basic training
python train.py --config=yolact_edge_config --batch_size=8

# Resume from interruption
python train.py --config=yolact_edge_config --resume=interrupt

# Transfer learning from COCO weights
python train.py --config=yolact_edge_config --resume=weights/yolact_base_54_800000.pth --yolact_transfer

# Multi-GPU training
python train.py --config=yolact_edge_config --batch_size=16 --num_gpus=2
```

### Evaluation/Inference
```bash
# Evaluate on validation set
python eval.py --trained_model=weights/yolact_edge_54_800000.pth

# Inference on image
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --image=input.jpg:output.jpg

# Inference on video
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --video=input.mp4

# Disable TensorRT optimization
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --disable_tensorrt

# Run benchmarks
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --benchmark
```

### Testing
No automated test suite exists. Validation is done through:
- Model evaluation on validation datasets
- Visual inspection using `--display` flag
- Performance benchmarking with `--benchmark`

## Architecture Overview

### Core Structure
- **yolact_edge/yolact.py**: Main YOLACT model implementation
- **yolact_edge/backbone.py**: Neural network backbones (ResNet, MobileNet, etc.)
- **yolact_edge/data/config.py**: All model and dataset configurations
- **yolact_edge/layers/**: Network modules (FPN, Protonet, etc.)
- **yolact_edge/utils/**: Utilities including augmentation, NMS, and TensorRT conversion

### Configuration System
The project uses a configuration-driven approach where models are defined as Config objects in `config.py`:
- Base configs: `yolact_base_config`, `yolact_edge_config`
- Video configs: `yolact_edge_vid_config` (with temporal features)
- Dataset configs: `coco2017_dataset`, `coco2017_person_dataset`, `youtube_vis_dataset`

### Key Components
1. **Backbone Network**: Feature extraction (ResNet50/101, MobileNetV2, DarkNet)
2. **FPN**: Feature Pyramid Network for multi-scale features
3. **Protonet**: Generates prototype masks
4. **Prediction Heads**: Classification, box regression, and mask coefficients
5. **NMS**: Non-maximum suppression (Cython-optimized)
6. **TensorRT Integration**: Automatic model optimization for edge deployment

### Person-Only Configuration
The current branch uses `coco2017_person_dataset` which:
- Uses SAMA-COCO annotations at `./data/sama-coco/`
- Contains only person class annotations
- Configured in `yolact_edge_config` at line 744-745

### Model Weights
Pre-trained weights are stored in `./weights/` directory. The project expects:
- Base YOLACT weights for transfer learning
- YOLACT Edge specific weights
- TensorRT optimized models (generated automatically)

## Development Tips

### Adding New Configurations
1. Define dataset in `config.py` following existing patterns
2. Create model config by copying and modifying existing configs
3. Update `num_classes` to match dataset classes + 1 (background)

### Debugging Training
- Use `--save_interval` to control checkpoint frequency
- Monitor with TensorBoard: logs saved to `../../logs/`
- Check GPU memory with smaller `--batch_size` if OOM

### Working with Videos
- Use video-specific configs ending with `_vid`
- These include temporal features for consistency
- Adjust `--eval_stride` for speed vs accuracy tradeoff

### TensorRT Optimization
- Models are automatically converted on first run
- INT8 calibration uses images from dataset's `calib_images` folder
- Use `--use_fp16_tensorrt` for FP16 precision instead of INT8
- Cached TensorRT engines stored alongside model weights