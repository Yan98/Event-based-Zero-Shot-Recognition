# EZSR: Event-based Zero-Shot Recognition

This repository contains the PyTorch code for our paper "EZSR: Event-based Zero-Shot Recognition".

> [paper]() | [arxiv]() | [project page](https://yan98.github.io/EZSR/)

**The inference code and pre-trained models will come soon**


## Introduction
This paper studies zero-shot object recognition using event camera data. Guided by CLIP, which is pre-trained on RGB images, existing approaches achieve zero-shot object recognition by maximizing embedding similarities between event data encoded by an event encoder and RGB images encoded by the CLIP image encoder. Alternatively, several methods learn RGB frame reconstructions from event data  for the CLIP image encoder. However, these approaches often result in suboptimal zero-shot performance.

This study develops an event encoder without relying on additional reconstruction networks. We theoretically analyze the performance bottlenecks of previous approaches: global similarity-based objective (i.e., maximizing the embedding similarities) cause semantic misalignments between the learned event embedding space and the CLIP text embedding space due to the degree of freedom. To mitigate the issue, we explore a scalar-wise regularization strategy. Furthermore, to scale up the number of events and RGB data pairs for training, we also propose a pipeline for synthesizing event data from static RGB images.

Experimentally, our data synthesis strategy exhibits an attractive scaling property, and our method achieves superior zero-shot object recognition performance on extensive standard benchmark datasets, even compared with past supervised learning approaches. For example, we achieve 47.84% zero-shot accuracy on the N-ImageNet dataset. 
## Framework

<div align=center>
<img src="asset/model.png", width=280/>
</div>

## Overview

<div align=center>
<img src="asset/overview.png", width=280/>
</div>

## Heatmap w.r.t to Text

<div align=center>
<img src="asset/heatmap.png", width=600/>
</div>


## Requirement

Please refer to [requirements.txt](./requirements.txt).

## How to run

```bash
python main.py
```

## How to get the dataset

```bash
python dataset.py
```

## Citation

```

@misc{yang2023eventcameradatadense,
      title={Event Camera Data Dense Pre-training}, 
      author={Yan Yang and Liyuan Pan and Liu Liu},
      year={2023},
      eprint={2311.11533},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.11533}, 
}



