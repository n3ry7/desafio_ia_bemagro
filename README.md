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

To 

### Dependencies


### Description


### Environments Variables


### Tests

