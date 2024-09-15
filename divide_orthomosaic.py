import argparse
from dataset_handler.image_splitter import ImageSplitter

def main():
    parser = argparse.ArgumentParser(description='Divide an orthomosaic into tiles')
    parser.add_argument('--input', required=True, help='Path to the orthomosaic file')
    parser.add_argument('--output', required=True, help='Path to the output directory')
    args = parser.parse_args()

    splitter = ImageSplitter(args.input, args.output)
    splitter.apply()

if __name__ == '__main__':
    main()
