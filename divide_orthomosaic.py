import argparse
import os
import shutil
from dataset_handler.image_splitter import ImageSplitter
from dataset_handler.preprocess_raw_tiles import PreprocessRawTiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Divide an orthomosaic into tiles')
    parser.add_argument('--input', required=True, help='Path to the orthomosaic file')
    parser.add_argument('--output', required=True, help='Path to the output directory')
    args = parser.parse_args()

    splitter = ImageSplitter(args.input, output_dir='dataset/raw_tiles')
    splitter.apply()

    # Preprocess raw tiles for image binarization
    preprocess_raw_tiles = PreprocessRawTiles(input_dir='dataset/raw_tiles', output_dir=args.output)
    preprocess_raw_tiles.apply()

    # Delete the 'dataset/raw_tiles' directory
    if os.path.exists('dataset/raw_tiles'):
        shutil.rmtree('dataset/raw_tiles')
