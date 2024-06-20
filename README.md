# Domain-Separation-Graph-Neural-Networks-for-Saliency-Object-Ranking

## Installation
Our code is primarily based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please refer to the [MMDetection Installation](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions.

## Dataset
Download the [ASSR Dataset](https://github.com/SirisAvishek/Attention_Shift_Ranks) and [IRSR Dataset](https://github.com/dragonlee258079/Saliency-Ranking).

## Training
### ASSR Dataset
For resnet-50 backbone model:
```bash
bash ./tools/dist_train.sh configs/mask2former_sor/mask2former_sor_r50_assr.py --num_gpus --load-from pertrained_model_path
```
For swin-L backbone model:
```bash
bash ./tools/dist_train.sh configs/mask2former_sor/mask2former_sor_swin-l-int21k_assr.py --num_gpus --load-from pertrained_model_path
```

### IRSR Dataset
For resnet-50 backbone model:
```bash
bash ./tools/dist_train.sh configs/mask2former_sor/mask2former_sor_r50_irsr.py --num_gpus --load-from pertrained_model_path
```
For swin-L backbone model:
```bash
bash ./tools/dist_train.sh configs/mask2former_sor/mask2former_sor_swin-l-int21k_irsr.py --num_gpus --load-from pertrained_model_path
```

## Citation
    @InProceedings{Wu_2024_CVPR,
        author    = {Wu, Zijian and Lu, Jun and Han, Jing and Bai, Lianfa and Zhang, Yi and Zhao, Zhuang and Song, Siyang},
        title     = {Domain Separation Graph Neural Networks for Saliency Object Ranking},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2024},
        pages     = {3964-3974}
    }
