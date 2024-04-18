# An Efﬁcient Single Image De-Raining Model With Decoupled Deep Networks
[Wencheng Li](https://scholar.google.com/citations?hl=en&user=P-sQphEAAAAJ), [Gang Chen](https://scholar.google.com/citations?hl=en&user=7GwIDigAAAAJ), [Yi Chang](https://scholar.google.com/citations?user=I1nZ67YAAAAJ&hl=en&oi=sra)

[Paper](https://ieeexplore.ieee.org/abstract/document/10336721)

> **Abstract:** *Single image de-raining is an emerging paradigm for many outdoor computer vision applications since rain streaks can significantly degrade the visibility and render the function compromised. The introduction of deep learning (DL) has brought about substantial advancement on de-raining methods. However, most existing DL-based methods use single homogeneous network architecture to generate de-rained images in a general image restoration manner, ignoring the discrepancy between rain location detection and rain intensity estimation. We find that this discrepancy would cause feature interference and representation ability degradation problems which significantly affect de-raining performance. In this paper, we propose a novel heterogeneous de-raining architecture aiming to decouple rain location detection and rain intensity estimation (DLINet). For these two subtasks, we provide dedicated network structures according to their differential properties to meet their respective performance requirements. To coordinate the decoupled subnetworks, we develop a high-order collaborative network learning the dynamic inter-layer interactions between rain location and intensity. To effectively supervise the decoupled subnetworks during training, we propose a novel training strategy that imposes task-oriented supervision using the label learned via joint training. Extensive experiments on synthetic datasets and real-world rainy scenes demonstrate that the proposed method has great advantages over existing state-of-the-art methods.* 

## Installation
The model is built in PyTorch 1.9.0 and tested on Ubuntu 18.04 environment (Python3.7, CUDA10.2).
The model is trained with 1 NVIDIA V100 GPU.

For installing, follow these intructions
```
conda install pytorch=1.9.0 torchvision=0.10.0
pip install numpy imageio scikit-image scipy matplotlib tqdm
```

## Training
### Synthetic datasets
*taking training Rain200H as an example*:
1. Download Rain200H and put it into the folder "./dataset/rain200H",  then the content is just like: 

    "./dataset/rain200H/train/rain/\*.png"

    "./dataset/rain200H/train/norain/\*.png"

    "./dataset/rain200H/test/rain/\*.png"
    
    "./dataset/rain200H/test/norain/\*.png"

2.  Begining training:
```
$ cd ./src/ 
$ python main.py --mode first_stage --num_mask 10 --num_msablock 15 --epochs 300 --batch_size 20 --patch_size 100 --dataset rain200H --data_range 1-1800/1-200 --lr 2e-4 --n_threads 6 --GPU_id cuda:0 
$ python main.py --mode second_stage --num_mask 10 --num_msablock 15 --epochs 300 --batch_size 16 --patch_size 100 --dataset rain200H --data_range 1-1800/1-200 --lr 2e-4 --n_threads 6 --GPU_id cuda:0
```
### spadata
1. Download spadata and put it into the folder "./dataset/spadata"
2. Download the index file of spadata from [Google Drive](https://drive.google.com/drive/folders/1-flV8M3V2lp2_BK6sPmcJsvXOc_Os7lf?usp=sharing), and and put them into "./src/data/".
2. Use ./src/data/srdata_spadata.py instead of ./src/data/srdata.py, you can simply rename the former as 'srdata.py'.
3. Begining training:
```
$ cd ./src/ 
$ python main.py --mode first_stage --num_mask 10 --num_msablock 15 --epochs 300 --batch_size 20 --patch_size 100 --dataset rain200H --data_range 1-638492/1-200 --lr 2e-4 --n_threads 6 --GPU_id cuda:0 
$ python main.py --mode second_stage --num_mask 10 --num_msablock 15 --epochs 300 --batch_size 16 --patch_size 100 --dataset rain200H --data_range 1-638492/1-1000 --lr 2e-4 --n_threads 6 --GPU_id cuda:0
```

## Evaluation
Please download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1x2IhV6G5IeoH4bkC52S5T5CS3YxpIohv?usp=sharing), and put them into respective folder.

For example, the fold for rain200H should be like: "./experiment/rain200H/stage2nd_Save/model"

```
$ cd ./src/
$ python main.py --mode second_stage --num_mask 10 --num_msablock 15 --dataset rain200H  --ext img  --data_range 1-1800/1-200 --pre_train ../experiment/rain200H/stage2nd_Save/model/model_best.pt --test_only --save_results
```

## Citation
If you use DLINet, please consider cite as follows:

    @ARTICLE{Li2023DLINet,
        author={Li, Wencheng and Chen, Gang and Chang, Yi},
        journal={IEEE Transactions on Image Processing}, 
        title={An Efficient Single Image De-Raining Model With Decoupled Deep Networks}, 
        year={2024},
        volume={33},
        pages={69-81}
    }

## Contact

If you have any question, please email `lwc1577148518@gmail.com`.

## Acknowledgements
This code is built on [JORDER-E](https://github.com/flyywh/JORDER-E-Deep-Image-Deraining-TPAMI-2019-Journal). We thank the authors for sharing their codes.
