# APPLIED MACHINE LEARNING SYSTEM ELEC0134 22/23 REPORT

This is the project researching super resolution (SR) based on DIV2K dataset and Set14 dataset.  

## Organization

- **/A**: Contains the models that are trained by the bicubic downscaled dataset in DIV2K.
- **/B**: Contains the models that are trained by the unknown downscaled dataset in DIV2K.
- **/Datasets**: Initially contains the validation dataset Set14. If training process starts, the DIV2K dataset will be automatically downloaded into this directory.

## Files

- **main.py**: Entry point for the application. There are few functions in the code containing training, evaluation, viualization and so on ,and they are seperated into few blocks but are annotated. If user need to run one of the block, just simply cancel the annotation. The description of every block is written above the corresponding code.
- **config.py**: This file is able to configure the training process, which includes the downscale factor, downscale way and file path.
- **data_preprocessing.py**: code for data pre-processing.
- **load_data.py**: code for downloading dataset and loading dataset.
- **model.py**: code of the models
- **train.py**: code for the training process
- **visualization.ipynb**: jupyter notebook file for visualizing the results, only for report purpose.


## Packages

The experiment environment is based on CUDA 12.2

Ensure you have the following packages installed before running the code:

```bash
pip install -r requirements.txt