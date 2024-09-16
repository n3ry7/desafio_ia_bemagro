import argparse
from plant_segmenter.train import UNetTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a U-Net model for plant segmentation")
    parser.add_argument("--rgb", type=str, required=True, help="Path to the directory containing the RGB images")
    parser.add_argument("--groundtruth", type=str, required=True, help="Path to the directory containing the ground truth segmented images")
    parser.add_argument("--modelpath", type=str, required=True, help="Path to save the trained model")
    args = parser.parse_args()

    trainer = UNetTrainer(
        train_img_dir=args.rgb,
        train_mask_dir=args.groundtruth,
        load_model=False,
        model_path=args.modelpath,
    )
    trainer.train()
