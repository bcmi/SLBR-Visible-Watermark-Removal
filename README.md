# Visible Watermark Removal via Self-calibrated Localization and Background Refinement
---

## Introduction
This is the official code of the following paper:

> 
> **Visible Watermark Removal via Self-calibrated Localization and Background Refinement**[[1]](#reference)
> <br>Jing Liang<sup>1</sup>, Li Niu<sup>1</sup>, Fengjun Guo<sup>2</sup>, Teng Long<sup>2</sup> and Liqing Zhang<sup>1</sup>
> <br><sup>1</sup>MoE Key Lab of Artificial Intelligence, Shanghai Jiao Tong University
> <br><sup>2</sup>INTSIG<br>
([ACM MM 2021](https://arxiv.org/pdf/2104.09453.pdf) | [Bibtex](#citation))


### SLBR Network
Here is our proposed **SLBR**(**S**elf-calibrated **L**ocalization and **B**ackground **R**efinement). Top row depicts the whole framework of SLBR and bottom row elaborates the details of our proposed three modules.
<div  align="center"> 
<img src="figs/framework.png" width = "100%" height = "100%" alt="Some examples of inharmonious region" align=center />
</div>
<div  align="center"> 
<img src="figs/Submodules.png" width = "100%" height = "100%" alt="Some examples of inharmonious region" align=center />
</div>


## Quick Start
### Install
- Install PyTorch>=1.0 following the [official instructions](https://pytorch.org/)
- git clone https://github.com/bcmi/SLBR-Visible-Watermark-Removal.git
- Install dependencies: pip install -r requirements.txt

### Data Preparation
In this paper, we conduct all of the experiments on the latest released dataset [CLWD](https://drive.google.com/file/d/17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij/view?usp=sharing)[[2]](#reference) and LVW[[3]](#reference). You can contact the authors of LVW to obtain the dataset.



### Train and Test
We provide a example of training and a test bash respectively:```scripts/train.sh```, ```scripts/test.sh``` 

Please specify the checkpoint save path in ```--checkpoint``` and dataset path in```--dataset_dir```.

## Visualization Results
We also show some qualitative comparision with state-of-art methods:

<div  align="center"> 
<img src="figs/bg_comparison.png" width = "90%" height = "90%" alt="Some examples of inharmonious region" align=center />
</div>


## **Acknowledgements**
Part of the code is based upon the previous work [SplitNet](https://github.com/vinthony/deep-blind-watermark-removal)[[4]](#reference).

## Citation
If you find this work or code is helpful in your research, please cite:
````
@article{liang2021visible,
  title={Visible Watermark Removal via Self-calibrated Localization and Background Refinement},
  author={Liang, Jing and Niu, Li and Guo, Fengjun and Long, Teng and Zhang, Liqing},
  journal={arXiv preprint arXiv:2108.03581},
  year={2021}
}
````
## Reference
[1] Visible Watermark Removal via Self-calibrated Localization and Background Refinement. Jing Liang, Li Niu, Fengjun Guo, Teng Long and Liqing Zhang. 2021. In *Proceedings of the 29th ACM International Conference on Multimedia*. [download](https://arxiv.org/pdf/2104.09453.pdf)

[2] WDNet: Watermark-Decomposition Network for Visible Watermark Removal. 2021. Liu, Yang and Zhu, Zhen and Bai, Xiang. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision.*

[3] Danni Cheng, Xiang Li, Wei-Hong Li, Chan Lu, Fake Li, Hua Zhao, and WeiShi Zheng. 2018. Large-scale visible watermark detection and removal with deep convolutional networks. In Chinese Conference on Pattern Recognition and Computer Vision. 27–40.

[4] Xiaodong Cun and Chi-Man Pun. 2020. Split then Refine: Stacked Attentionguided ResUNets for Blind Single Image Visible Watermark Removal. arXiv preprint arXiv:2012.07007 (2020).
