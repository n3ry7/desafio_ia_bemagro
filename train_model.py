import argparse
from plant_segmenter.u_net_trainer import UNetTrainer
from dataset_handler.data_splitter import DataSplitter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a U-Net model for plant segmentation")
    parser.add_argument("--rgb", type=str, required=True, help="Path to the directory containing the RGB images")
    parser.add_argument("--groundtruth", type=str, required=True, help="Path to the directory containing the ground truth segmented images")
    parser.add_argument("--modelpath", type=str, required=True, help="Path to save the trained model")
    args = parser.parse_args()

    # Split the dataset into training and validation sets
    data_splitter = DataSplitter(args.rgb, args.groundtruth)
    train_image_files, train_mask_files, val_image_files, val_mask_files = data_splitter.apply()

    # Train the U-Net model
    trainer = UNetTrainer(
        train_img_dir=data_splitter.train_image_dir,
        train_mask_dir=data_splitter.train_mask_dir,
        val_img_dir=data_splitter.val_image_dir,
        val_mask_dir=data_splitter.val_mask_dir,
        load_model=False,
        model_path=args.modelpath,
    )
    trainer.train()
