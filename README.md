# gazeNet: End-to-end eye-movement event detection with deep neural networks

Cite as:
```sh
@article{zemblys2018gazeNet,
  title={gazeNet: End-to-end eye-movement event detection with deep neural networks},
  author={Zemblys, Raimondas and Niehorster, Diederick C and Holmqvist, Kenneth},
  journal={Behavior research methods},
  year={2018},
}
```
gazeNet was developed on linux using Python 2.7 and PyTorch 0.2.0_4. Other required packages: `numpy`, `pandas`, `tensorboard`, `Levenshtein`, `scikit-learn`

## Training a new model
To train a new gazeNet model you will need your own coded eye-movement data or alternatively you can use Lund2013 dataset. 

If you use your own dataset, convert it to ETData format and copy training and validation sets to separate folders in `./logdir/MODEL_DIR/data`. See [Config file](# Config file) below for further setup. Also check `./utils_lib/data_prep/tt_split.py` for an example how to convert your data to ETData format.


### Preparing Lund2013 dataset
In case you want to use Lund2013 dataset:

- navigate to `./etdata` and run  `git clone https://github.com/richardandersson/EyeMovementDetectorEvaluation.git`. This will download manually coded eye-movement data together with the software used to code that data.

- run `./utils_lib/data_prep/tt_split.py`. Script will convert data from matlab to ETData format and save it to `./etdata/lund2013_npy`. In addition it will split dataset into training, validation and testing sets and save it as pickled lists to `./etdata/gazeNet_data`. 

- from `./etdata/gazeNet_data` copy `data.unpaired_clean.pkl` and `data.val_clean.pkl` to `./logdir/MODEL_DIR/data`. 

- from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1476449.svg)](https://doi.org/10.5281/zenodo.1476449)  download `data.gen.pkl` and place it to `./logdir/MODEL_DIR/data` 

### Training gazeNet
To train a new model run `train.py`. Script takes the following arguments:
```sh
  --model_dir MODEL_DIR
                        Directory in which to store the logging
  --num_workers NUM_WORKERS
                        Number of workers used in dataloading
  --num_epochs NUM_EPOCHS
                        Number of epochs to train
  --seed SEED           seed

```
For example run:
`python train.py --model_dir MODEL_DIR --num_epochs 20`

This will train a model for 20 epochs and save it to `./logdir/MODEL_DIR/models`. `MODEL_DIR` directory needs to contain `config.json` file and `data` directory that stores training and validation data.

#### Config file
Config file is a json file that describes model architecture, training parameters, and datasets. For an example file see `./logdir/model_final/config.json`.

The following variables in `config.json` define training and validatation datasets. It can be a pickled lists of ETData arrays or folders with numpy files in ETData format (the later is experimental and was not extensively  tested). **NOTE** that current version of the code uses 2 validation sets!
```json
"data_train": ["data.gen.pkl"], 
"data_val": ["data.val_clean.pkl"], 
"data_train_gen": ["data.unpaired_clean.pkl"]
```

Training data that was used in the paper can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1476449.svg)](https://doi.org/10.5281/zenodo.1476449), while validation datasets can be generated using `./utils_lib/data_prep/tt_split.py` script (see [Preparing Lund2013 dataset](# Preparing Lund2013 dataset) above ).


*TODO: decribe other parameters*

## Running gazeNet
To run gazeNet use `run_gazeNet.py`. Script takes the following parameters:

```sh
positional arguments:
  root                  Root for datasets
  dataset               Dataset

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Directory in which to store the logging
  --model_name MODEL_NAME
                        Model
  --num_workers NUM_WORKERS
                        Number of workers used in dataloading
  --save_pred           Whether to save predictions; not currently used
```

For example to parse `lund2013` dataset run:
`python run_gazeNet.py etdata lund2013_npy`

By default the script uses the pretrained model - `logdir/model_final/models/gazeNET_0004_00003750.pth.tar` - the model that was used in the paper. The output, together with scanpaths and time series plots, will be saved to `./etdata/lund2013_npy_gazeNet`. 

To use your own trained model run:
`python run_gazeNet.py etdata lund2013_npy --model_dir MODEL_DIR --model_name MODEL_NAME`
where `MODEL_DIR` and `MODEL_NAME` are the directory and the file name of your model.
