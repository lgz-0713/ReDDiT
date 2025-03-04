
<h2 align="center"> Efficient Diffusion as Low Light Enhancer (CVPR 2025)</h2>



## Get Started

### Dependencies and Installation

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

### Data Preparation

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

### Testing

Note: Following LLFlow and KinD, we have also adjusted the brightness of the output image produced by the network, based on the average value of Ground Truth (GT). ``It should be noted that this adjustment process does not influence the texture details generated; it is merely a straightforward method to regulate the overall illumination.`` Moreover, it can be easily adjusted according to user preferences in practical applications.

You can also refer to the following links to download the [pretrained model](https://drive.google.com/drive/folders/1XUU26LJivbl9pfx1XlFI4Uf7Oi4bTqqo?usp=sharing) and put it in the following folder:

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
### Testing on unpaired data

```
python test_unpaired.py  --config config/test_unpaired.json --input unpaired_image_folder
```

You can use any one of these three pre-trained models, and employ different sampling steps to obtain visual-pleasing results by modifying these terms in the 'test_unpaired.json'.



### Training

```
bash train.sh
```


## Acknowledgement

Our code is built upon [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). Thanks to the contributors for their great work.
