# AI Challenge 

## Usage

Use the following commands:

```bash
 source .venv/bin/activate
```



```bash
python divide_orthomosaic.py --input orthomosaic.tif --output dataset/images
```

```bash
python binarize_images.py --input dataset/images --output dataset/binary_masks
```

```bash
python train_model.py --rgb dataset/images --groundtruth dataset/binary_masks --modelpath plant_segmenter/weights/u_net_weights.pth
```

```bash
python model_inference.py --rgb pictures/1.jpg --modelpath plant_segmenter/weights/u_net_weights.pth --output pictures/results/1.png
```


## Installation using Makefile

Use the command
```bash
make install
```

To create a python virtual environment at .venv and install all requirements.

### Dependencies

This project used python 3.12.2 and the packages are listed on requirements/requirements.txt

### Description

This project  creates a dataset from a single tif image and train a U-net model for instance segmentation.


### Environments Variables

The following environment variables could be defined in a .env file to specify training parameters:

- **LEARNING_RATE** (*float: LEARNING_RATE* **default=1e-4**): The learning rate for the optimizer.
- **DEVICE** (*str: DEVICE* **default=cuda if available, else cpu**): The device to use for training and inference ('cuda' or 'cpu').
- **BATCH_SIZE** (*int: BATCH_SIZE* **default=32**): The batch size for training and validation.
- **NUM_EPOCHS** (*int: NUM_EPOCHS* **default=4**): The number of epochs to train the model.
- **NUM_WORKERS** (*int: NUM_WORKERS* **default=4**): The number of workers to use for data loading.
- **IMAGE_HEIGHT** (*int: IMAGE_HEIGHT* **default=256**): The height of the input images.
- **IMAGE_WIDTH** (*int: IMAGE_WIDTH* **default=256**): The width of the input images.
- **PIN_MEMORY** (*bool: PIN_MEMORY* **default=True**): Whether to use pin memory for data loading.
- **LOAD_MODEL** (*bool: LOAD_MODEL* **default=False**): Whether to load a pre-trained model.
- **MODEL_PATH** (*str: MODEL_PATH* **default=plant_segmenter/weights/u_net_weights.pth**): Path to save trained weights.
- **TRAIN_IMG_DIR** (*str: TRAIN_IMG_DIR* **default=dataset/train_images**): The directory containing the training images.
- **TRAIN_MASK_DIR** (*str: TRAIN_MASK_DIR* **default=dataset/train_masks**): The directory containing the training masks.
- **VAL_IMG_DIR** (*str: VAL_IMG_DIR* **default=dataset/val_images**): The directory containing the validation images.
- **VAL_MASK_DIR** (*str: VAL_MASK_DIR* **default=dataset/val_masks**): The directory containing the validation masks.
