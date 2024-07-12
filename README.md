# Beyond Image Prior: Embedding Noise Prior into Conditional Denoising Transformer (Condformer)
This repository is for Condformer introduced in the following paper

Yuanfei Huang and Hua Huang*, "Beyond Image Prior: Embedding Noise Prior into Conditional Denoising Transformer",arXiv.
[paper]()
## Dependenices
* python 3.10
* pytorch == 2.0.0
* NVIDIA GPU + CUDA

## Models
Download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1C4lhvT0FdY416pROf5Ckmza6-I9a42YQ?usp=drive_link)

## Data preparing
Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) datasets into the path "Datasets/Train/DF2K" for synthetic image denoising task.
Download [SIDD-Medium](https://abdokamel.github.io/sidd/) datasets into the path "Datasets/Train/SIDD_Medium_Srgb" for real image denoising task.


## Settings (option.py)
For synthetic image denoising:
* '-data_train' == ['DF2K', 'WED', 'BSD']
* '-data_test' == ['CBSD68', 'Kodak24', 'Urban100']
* '-n_train' == [3450, 4744, 400]

For real image denoising:
* '-data_train' == ['SIDD_Medium_Srgb']
* '-data_test' == ['SIDD']
* '-n_train' == [320]

## Train
```bash
python main.py --train 'train'
```
## Test
```bash
python main.py --train 'test'
```
## Calculate complexity
```bash
python main.py --train 'complexity'
```

## Citation
```

```
