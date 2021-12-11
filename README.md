## PyTorch implementation of "Light Field Image Super-Resolution Using Deformable Convolution", <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286855">TIP 2021</a><br>

## News:
* **2021-12-11**: We recommend our newly-released repository [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR) for the implementation of our LF-DFnet. [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR) is an open-source and easy-to-use toolbox for LF image SR. The codes of several milestone methods (e.g., LFSSR, LF-ATO, LF-InterNet, LF-DFnet) have been implemented (retrained) in a unified framework in [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR).
<br>

## Network Architecture:
<p align="center"> <img src="https://raw.github.com/YingqianWang/LF-DFnet/master/Figs/Network.jpg" width="95%"> </p><br>

## Codes and Models:

### Directly Download the Results of LF-DFnet:
**We share the super-resolved LF images generated by our LF-DFnet on all the 5 datasets for 4xSR. Then, researchers can compare their algorithms to our LF-DFnet without performing inference. Results are available at [Baidu Drive](https://pan.baidu.com/s/1SfuH3UEb8FbIGXoijMb6kw) (Key: nudt).**
<br><br>

### Datasets:
**We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for both training and test. Please first download our dataset via [Baidu Drive](https://pan.baidu.com/s/144kg-c94EIJrzSkd-wxK9A) (key:nudt) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), then place the 5 datasets to the folder `./Datasets/`.**<br><br>

### Requirement:
* **PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=9.0.**
* **Matlab (For training/test data generation and result image generation)**<br><br>

### Compile DCN:
* **Cd to `code/dcn`.**
* **For Windows users, run `cmd make.bat`. For Linux users, run bash `bash make.sh`. The scripts will build DCN automatically and create some folders.
    See `test.py` for example usage.**
<br><br>

### Train:
* **Run `GenerateTrainingData.m` to generate training data. The generated data will be saved in `./Data/TrainData_UxSR_AxA/` (U=2,4; A=3,5,7,9).**
* **Run `train.py` to perform network training. Note that, the training settings in `train.py` should match the generated training data. Checkpoint will be saved to `./log/`.**
<br><br>

### Test on our datasets:
* **Run `GenerateTestData.m` to generate input LFs of the test set. The generated data will be saved in `./Data/TestData_UxSR_AxA/` (U=2,4; A=3,5,7,9).**
* **Run `test.py` to perform network inference. The PSNR and SSIM values of each dataset will be printed on the screen.**
* **Run `GenerateResultImages.m` to convert '.mat' files in `./Results/` to '.png' images to `./SRimages/`.**
<br><br>

## Results in Our Paper:
### Quantitative Results:
<p align="center"> <img src="https://raw.github.com/YingqianWang/LF-DFnet/master/Figs/Quantitative.jpg" width="100%"> </p>

### Visual Comparisons:
<p align="center"> <img src="https://raw.github.com/YingqianWang/LF-DFnet/master/Figs/2xSR.jpg" width="100%"> </p>
<p align="center"> <img src="https://raw.github.com/YingqianWang/LF-DFnet/master/Figs/4xSR.jpg" width="100%"> </p>

### Efficiency:
<p align="center"> <img src="https://raw.github.com/YingqianWang/LF-DFnet/master/Figs/Efficiency.jpg" width="50%"> </p>

### Performance w.r.t. Perspectives:
<p align="center"> <img src="https://raw.github.com/YingqianWang/LF-DFnet/master/Figs/PwrtP.jpg" width="100%"> </p>

### Performance w.r.t. Baseline Lengths:
<p align="center"> <img src="https://raw.github.com/YingqianWang/LF-DFnet/master/Figs/PwrtB.jpg" width="60%"> </p>

### Benefits to Depth Estimation (i.e., Angular Consistency):
<p align="center"> <img src="https://raw.github.com/YingqianWang/LF-DFnet/master/Figs/Depth.jpg" width="100%"> </p>

### Performance on Real LFs:
<p align="center"> <img src="https://raw.github.com/YingqianWang/LF-DFnet/master/Figs/realSR.jpg" width="70%"> </p>


## Citiation
**If you find this work helpful, please consider citing the following paper:**
```
@article{LF-DFnet,
  author  = {Wang, Yingqian and Yang, Jungang and Wang, Longguang and Ying, Xinyi and Wu, Tianhao and An, Wei and Guo, Yulan},
  title   = {Light Field Image Super-Resolution Using Deformable Convolution},
  journal = {IEEE Transactions on Image Processing},
  volume  = {30),
  pages   = {1057-1071},
  year    = {2021},
}
```
<br><br>

## Acknowledgement
**The DCN part of our code is referred from [DCNv2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0) and [D3Dnet](https://github.com/XinyiYing/D3Dnet). We thank the authors for sharing their codes.**
<br><br>

## Contact
**Any question regarding this work can be addressed to wangyingqian16@nudt.edu.cn.**
