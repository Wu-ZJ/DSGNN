# Domain-Separation-Graph-Neural-Networks-for-Saliency-Object-Ranking
Code release for the CVPR 2024 paper [Domain Separation Graph Neural Networks for Saliency Object Ranking](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Domain_Separation_Graph_Neural_Networks_for_Saliency_Object_Ranking_CVPR_2024_paper.html)
<img src="https://github.com/Wu-ZJ/DSGNN/blob/main/resources/main.png"/>

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
</details>

## Testing
### ASSR Dataset
For resnet-50 backbone model:
```bash
bash ./tools/dist_test.sh configs/mask2former_sor/mask2former_sor_r50_assr.py model_path 1 --eval mae
```
For swin-L backbone model:
```bash
bash ./tools/dist_test.sh configs/mask2former_sor/mask2former_sor_swin-l-int21k_assr.py model_path 1 --eval mae
```

### IRSR Dataset
For resnet-50 backbone model:
```bash
bash ./tools/dist_test.sh configs/mask2former_sor/mask2former_sor_r50_irsr.py model_path 1 --eval mae
```
For swin-L backbone model:
```bash
bash ./tools/dist_test.sh configs/mask2former_sor/mask2former_sor_swin-l-int21k_irsr.py model_path 1 --eval mae
```

## Results
### Models

| Model | Dataset | Download |  
| :---: | :---: | :---: |  
| Pertrained-Res50 | COCO | [mask2former_r50_lsj_8x2_50e_coco](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth) |  
| Pertrained-SwinL | COCO | [mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic](https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth) |
| DSGNN-Res50 | ASSR | model \| [visualization results](链接：https://pan.baidu.com/s/1x4vP3M7uiPeAissD6x7lCQ) (9esz) |  
| DSGNN-SwinL | ASSR | model \| [visualization results](链接：https://pan.baidu.com/s/1x4vP3M7uiPeAissD6x7lCQ) (9esz) |
| DSGNN-Res50 | IRSR | 内容 |  
| DSGNN-SwinL | IRSR | 内容 |

### Visualization Results

## Citation
    @InProceedings{Wu_2024_CVPR,
        author    = {Wu, Zijian and Lu, Jun and Han, Jing and Bai, Lianfa and Zhang, Yi and Zhao, Zhuang and Song, Siyang},
        title     = {Domain Separation Graph Neural Networks for Saliency Object Ranking},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2024},
        pages     = {3964-3974}
    }
