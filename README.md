# DAME:Automatic detection of melanins and sebums from skin images using generative adversarial network
#### Lun Hu
#### Peng Zhou
----
### Folders
* `data` contains data processing related scripts.
* `datasets` contains pre-training script.
* `models` contains Pix2pix related deep learning model.
* `options` contains scripts that configure the relevant parameters.
* `util` contains frame application related scripts.
### Files
* `prepare.py` contains code about image gray and image enchancement.
* `test.py` contains related code to test.
* `train.py` contains related code to train.

### Usage
1. ` git clone` this project;

**note**:`ImageDataset` folder is test dataset;`SourceCodes` is model source code

2. `cd DAME/SourceCodes`(root directory),new folder named `image`, the directory structure is as follows:
place the **original image** in the jpg folder and place the **marked image** in the mark folder.
```
image
│   ├── finish
│   ├── origin
│   │   ├── black
│   │   │   ├── jpg
│   │   │   └── mark
│   │   └── oil
│   │       ├── jpg
│   │       └── mark 
```
3. run `python prepare.py` and it will generate image after process(DAME or guassian or CLAHE). the directory structure is as follows:
```
image
│   ├── finish
│   │   ├── black
│   │   │   ├── jpg
│   │   │   └── mark
│   │   └── oil
│   │       ├── jpg
│   │       └── mark
│   ├── origin
```
**note**: different processes correspond to different functions `DAME->prepare.py/gray`,`guassian->prepare.py/gaussian`,`CLAHE->prepare.py/clahe`

4. in the root directory, run `python datasets/combine_A_and_B.py --fold_A finish/black/jpg --fold_B finish/black/mark --fold_AB finish/black/ --no_multiprocessing`, It will generate the dataset required by the pix2pix model.
5. train model:`python train.py --dataroot finish/black --model pix2pix --name black`
6. test model:`python test.py --dataroot ./datasets/black/ --name black --model pix2pix`
