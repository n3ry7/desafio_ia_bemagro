from dataset_handler.binary_mask_generator import BinaryMaskGenerator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binarize Images')
    parser.add_argument('--input', required=True, help='Path to the input raw tiles directory')
    parser.add_argument('--output', required=True, help='Path to the output binary masks directory')
    args = parser.parse_args()

    # Generate binary masks
    binary_mask_generator = BinaryMaskGenerator(
        input_folder=args.input,
        output_folder_masks=args.output,
    )
    binary_mask_generator.apply()
