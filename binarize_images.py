from dataset_handler.binary_mask_generator import BinaryMaskGenerator
from dataset_handler.preprocess_raw_tiles import PreprocessRawTiles
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binarize Images')
    parser.add_argument('--input', required=True, help='Path to the input raw tiles directory')
    parser.add_argument('--output', required=True, help='Path to the output binary masks directory')
    args = parser.parse_args()

    # Preprocess raw tiles
    preprocess_raw_tiles = PreprocessRawTiles(input_dir=args.input, output_dir='dataset/images')
    preprocess_raw_tiles.apply()

    # Generate binary masks
    binary_mask_generator = BinaryMaskGenerator(
        input_folder="dataset/images",
        output_folder_masks=args.output,
        output_folder_plants="dataset/result_cluster_images"
    )
    binary_mask_generator.apply()
