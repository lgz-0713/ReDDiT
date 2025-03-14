

<div align=center>
  
# **[CVPR2025]** Efficient Diffusion as Low Light Enhancer

<p>
<a href='https://arxiv.org/abs/2410.12346'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://mqleet.github.io/ReDDiT_Project/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
</p>

</div>

## :fire: News

- [2025/03/04] We have released the training code and inference code! 🚀🚀
- [2025/02/27] ReDDiT has been accepted to CVPR 2025! 🤗🤗

## :memo: TODO

- [x] Training code
- [x] Inference code
- [x] CVPR Camera-ready Version
- [x] Project page

## :hammer: Get Started

### :mag: Dependencies and Installation

- Python 3.8
- Pytorch 1.11

1. Create Conda Environment

```
conda create --name ReDDiT python=3.8
conda activate ReDDiT
```

2. Install PyTorch

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Install Dependencies

```
cd ReDDiT
pip install -r requirements.txt
```

### :page_with_curl: Data Preparation

You can refer to the following links to download the datasets.

- [LOLv1](https://daooshee.github.io/BMVC2018website/)
- [LOLv2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)

Then, put them in the following folder:

<details open> <summary>dataset (click to expand)</summary>

```
├── dataset
    ├── LOLv1
        ├── our485
            ├──low
            ├──high
	├── eval15
            ├──low
            ├──high
├── dataset
   ├── LOLv2
       ├── Real_captured
           ├── Train
	   ├── Test
       ├── Synthetic
           ├── Train
	   ├── Test
```

</details>

### :blue_book: Testing

Note: Following LLFlow and KinD, we have also adjusted the brightness of the output image produced by the network, based on the average value of Ground Truth (GT). ``It should be noted that this adjustment process does not influence the texture details generated; it is merely a straightforward method to regulate the overall illumination.`` Moreover, it can be easily adjusted according to user preferences in practical applications.

You can also refer to the following links to download the [Checkpoints](https://drive.google.com/file/d/13_XM8nFxJc2IfUotC2_lJo9ATt0rcIyg/view?usp=sharing) and put it in the following folder:

```
├── checkpoints
    ├── lolv1_8step_gen.pth
    ├── lolv1_4step_gen.pth
    ├── lolv1_2step_gen.pth
    ......
```
To test the model using the ``sh test.sh`` command and modify the `n_timestep` and `time_scale` parameters for different step models. Here's a general outline of the steps:
```
"val": {
    "schedule": "linear",
                "n_timestep": 8,
                "linear_start": 1e-4,
                "linear_end": 2e-2,
                "time_scale": 64
}
```

```
"val": {
    "schedule": "linear",
                "n_timestep": 4,
                "linear_start": 1e-4,
                "linear_end": 2e-2,
                "time_scale": 128
}
```

```
"val": {
    "schedule": "linear",
                "n_timestep": 2,
                "linear_start": 1e-4,
                "linear_end": 2e-2,
                "time_scale": 256
}
```
### :blue_book: Testing on unpaired data

```
python test_unpaired.py  --config config/test_unpaired.json --input unpaired_image_folder
```

You can use any one of these three pre-trained models, and employ different sampling steps to obtain visual-pleasing results by modifying these terms in the 'test_unpaired.json'.



### :rocket: Training

```
bash train.sh
```

<a name="citation_and_acknowledgement"></a>
## :black_nib: Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @InProceedings{lan2024towards,
    title={Efficient Diffusion as Low Light Enhancer},
    author={Lan, Guanzhou and Ma, Qianli and Yang, Yuqi and Wang, Zhigang and Wang, Dong and Li, Xuelong and Zhao, Bin},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
    }
   ```


## :heart: Acknowledgement

Our code is built upon [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). Thanks to the contributors for their great work.
