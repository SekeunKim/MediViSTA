# MediViSTA: Medical Video Segmentation via Temporal Fusion SAM Adaptation for Echocardiography


[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 

This repo contains the code for our paper  <a href="https://arxiv.org/abs/2309.13539"> **MediViSTA: Medical Video Segmentation via Temporal Fusion SAM Adaptation for Echocardiography**  </a>.

![Overview of framework](method.png?raw=true "Overview of MeediViSTA framework")

> Despite achieving impressive results in general-purpose semantic segmentation with strong generalization on natural images, the Segment Anything Model (SAM) has shown less precision and stability in medical image segmentation. In particular, the original SAM architecture is designed for 2D natural images and is therefore not support to handle three-dimensional information, which is particularly important for medical imaging modalities that are often volumetric or video data. In this paper, we introduce MediViSTA, a parameter-efficient fine-tuning method designed to adapt the vision foundation model for medical video, with a specific focus on echocardiography segmentation. To achieve spatial adaptation, we propose a frequency feature fusion technique that injects spatial frequency information from a CNN branch. For temporal adaptation, we integrate temporal adapters within the transformer blocks of the image encoder. Using a fine-tuning strategy, only a small subset of pre-trained parameters is updated, allowing efficient adaptation to echocardiography data. The effectiveness of our method has been comprehensively evaluated on three datasets, comprising two public datasets and one multi-center in-house dataset. Our method consistently outperforms various state-of-the-art approaches without using any prompts. Furthermore, our model exhibits strong generalization capabilities on unseen datasets, surpassing the second-best approach by 2.15\% in Dice and 0.09 in temporal consistency. The results demonstrate the potential of MediViSTA to significantly advance echocardiography video segmentation, offering improved accuracy and robustness in cardiac assessment applications.


## Execution Instructions
- Envrionment Setting

```
pip install -r requirements.py
```
  
- Build Model
```
from models.segmentation.segment_anything import sam_model_registry
model, img_embedding_size = sam_model_registry[args.vit_type](args, image_size=args.img_size,
                                                num_classes=args.num_classes,
                                                chunk = chunk,
                                                checkpoint=args.resume, pixel_mean=[0., 0., 0.],
                                                pixel_std=[1., 1., 1.])
```

## Pretrained Model Chcekpoints
We employed pretrained SAM model to train our model. 
Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Citation

If you found MediViSTA-SAM useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

```bibtex
@article{kim2023medivista,
  title={MediViSTA-SAM: Zero-shot Medical Video Analysis with Spatio-temporal SAM Adaptation},
  author={Kim, Sekeun and Kim, Kyungsang and Hu, Jiang and Chen, Cheng and Lyu, Zhiliang and Hui, Ren and Kim, Sunghwan and Liu, Zhengliang and Zhong, Aoxiao and Li, Xiang and others},
  journal={arXiv preprint arXiv:2309.13539},
  year={2023}
}
```

## Acknowledgement
We thank MetaAI [(https://github.com/NVlabs/edm)](https://github.com/facebookresearch/segment-anything) for providing baseline method.