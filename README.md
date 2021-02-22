### PyTorch implementation of "Light Field Image Super-Resolution Using Deformable Convolution", <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286855">TIP 2021</a><br><br>

## Requirement
* **PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=9.0.**
* **Matlab (For training/test data generation and performance evaluation)**

## Datasets
* **We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for both training and test. Please first download our dataset via [Baidu Drive](https://pan.baidu.com/s/144kg-c94EIJrzSkd-wxK9A) (key:nudt), and place the 5 datasets to the folder `./Datasets/`.**

## Compile DCN

* **Cd to `code/dcn`.**
* **For Windows users, run `cmd make.bat`. For Linux users, run bash `bash make.sh`. The scripts will build DCN automatically and create some folders.
    See `test.py` for example usage.**


## Train
* **Run `GenerateTrainingData.m` to generate training data. The generated data will be saved in `./Data/TrainData_UxSR_AxA/` (U=2,4; A=3,5,7,9).**
* **Run `train.py` to perform network training. Note that, the training settings in `train.py` should match the generated training data. Checkpoint will be saved to `./log/`.**

## Test on our datasets
* **Run `GenerateTestData.m` to generate input LFs of the test set. The generated data will be saved in `./Data/TestData_UxSR_AxA/` (U=2,4; A=3,5,7,9).**
* **Run `test.py` to perform network inference. The PSNR and SSIM values of each dataset will be printed on the screen.**
* **Run `GenerateResultImages.m` to convert '.mat' files in `./Results/` to '.png' images to `./SRimages/`.**

## Test on your own LFs
**Will be released soon.**

## Citiation
**If you find this work helpful, please consider citing the following paper:**
```
@article{LF-DFnet,
  author  = {Wang, Yingqian and Yang, Jungang and Wang, Longguang and Ying, Xinyi and Wu, Tianhao and An, Wei and Guo, Yulan},
  title   = {Spatial-Angular Interaction for Light Field Image Super-Resolution},
  journal = {IEEE Transactions on Image Processing},
  volume  = {30),
  pages   = {1057-1071},
  year    = {2021},
}
```

## Acknowledgement
**The DCN part of our code is referred from [DCNv2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0) and [D3Dnet](https://github.com/XinyiYing/D3Dnet). We thank the authors for sharing their codes.**

## Contact
**Any question regarding this work can be addressed to wangyingqian16@nudt.edu.cn.**
