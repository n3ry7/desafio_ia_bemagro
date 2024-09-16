import cv2
import argparse
import numpy as np
import torch
from plant_segmenter.plant_segmenter import PlantSegmenter

def main():
    parser = argparse.ArgumentParser(description='Plant Segmentation Inference')
    parser.add_argument('--rgb', type=str, required=True, help='Path to the input RGB image')
    parser.add_argument('--modelpath', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--output', type=str, required=True, help='Path to save the segmented image')
    args = parser.parse_args()

    # Load the input image using OpenCV
    rgb_image = cv2.imread(args.rgb)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_image = rgb_image.astype(np.float32) / 255.0
    rgb_image = torch.from_numpy(rgb_image.transpose(2, 0, 1))

    # Load the plant segmenter
    plant_segmenter = PlantSegmenter(args.modelpath)

    # Perform inference
    segmented_image = plant_segmenter.segment(rgb_image)

    # Save the segmented image
    segmented_image = (segmented_image * 255).byte()
    cv2.imwrite(args.output, segmented_image.permute(1, 2, 0).byte().numpy())

if __name__ == '__main__':
    main()
