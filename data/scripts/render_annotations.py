#!/usr/bin/env python3
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools import mask as maskUtils
import random
import argparse

def decode_segmentation(segmentation, height, width):
    """Decode segmentation from COCO format to binary mask"""
    if isinstance(segmentation, list):
        # Check if it's a list of polygons
        if len(segmentation) > 0 and isinstance(segmentation[0], list):
            # Polygon format
            mask = np.zeros((height, width), dtype=np.uint8)
            for seg in segmentation:
                poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [poly], 1)
            return mask
    elif isinstance(segmentation, dict):
        # RLE format
        if 'counts' in segmentation and 'size' in segmentation:
            try:
                mask = maskUtils.decode(segmentation)
                return mask
            except:
                # If decoding fails, skip this segmentation
                return None
    return None

def render_annotations(image_path, annotations, output_path):
    """Render bounding boxes and segmentation masks on image"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    # Colors for different instances
    colors = plt.cm.rainbow(np.linspace(0, 1, len(annotations)))
    
    for ann, color in zip(annotations, colors):
        # Draw segmentation mask
        if 'segmentation' in ann and ann['segmentation']:
            mask = decode_segmentation(ann['segmentation'], height, width)
            if mask is not None:
                # Create colored mask
                colored_mask = np.zeros((height, width, 4))
                colored_mask[:,:,0] = color[0]
                colored_mask[:,:,1] = color[1]
                colored_mask[:,:,2] = color[2]
                colored_mask[:,:,3] = mask * 0.5  # 50% transparency
                ax.imshow(colored_mask)
        
        # Draw bounding box
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add annotation ID as label
            if 'id' in ann:
                ax.text(x, y-5, f"ID: {ann['id']}", color=color, 
                       fontsize=8, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    ax.set_title(f"Image: {Path(image_path).name}")
    ax.axis('off')
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Render COCO annotations on images')
    parser.add_argument('--num-samples', type=int, default=50, 
                        help='Number of sample images to render (default: 50)')
    parser.add_argument('--annotation-file', type=str, 
                        default='annotations/instances_val2017_person_only.json',
                        help='Path to annotation file')
    parser.add_argument('--image-dir', type=str, default='images/val2017',
                        help='Path to image directory')
    parser.add_argument('--output-dir', type=str, default='annotation_samples',
                        help='Output directory for rendered images')
    args = parser.parse_args()
    
    # Load annotations
    print(f"Loading annotations from {args.annotation_file}...")
    with open(args.annotation_file, 'r') as f:
        data = json.load(f)
    
    # Create image_id to annotations mapping
    image_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)
    
    # Create image_id to filename mapping
    id_to_info = {img['id']: img for img in data['images']}
    
    # Get images that have annotations
    annotated_images = list(image_to_anns.keys())
    print(f"Found {len(annotated_images)} images with annotations")
    
    # Select random images
    sample_size = min(args.num_samples, len(annotated_images))
    selected_ids = random.sample(annotated_images, sample_size)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nRendering {sample_size} sample images...")
    
    success_count = 0
    for i, img_id in enumerate(selected_ids):
        if img_id not in id_to_info:
            continue
            
        img_info = id_to_info[img_id]
        img_filename = img_info['file_name']
        img_path = Path(args.image_dir) / img_filename
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Get annotations for this image
        anns = image_to_anns[img_id]
        
        # Output filename
        output_path = output_dir / f"sample_{i+1:03d}_{img_filename}"
        
        print(f"[{i+1}/{sample_size}] Processing {img_filename} ({len(anns)} annotations)...")
        
        if render_annotations(img_path, anns, output_path):
            success_count += 1
            print(f"  Saved to: {output_path}")
    
    print(f"\nSuccessfully rendered {success_count} images")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()