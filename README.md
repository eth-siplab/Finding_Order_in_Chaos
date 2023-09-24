# Finding_Order_in_Chaos



## Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive Learning (NeurIPS 2023, Official Code)

Berken Utku Demirel, Christian Holz<br/>

<p align="center">
</p>

___________


<p align="center">
<img src="figs/teaser.gif" width="800">
</p>


---

> In this paper, we propose a novel data augmentation method for quasi-periodic time-series tasks that aims to connect intra-class samples together, and thereby find order in the latent space.
Our method builds upon the well-known mixup technique by incorporating a novel approach that accounts for the periodic nature of non-stationary time-series.
Also, by controlling the degree of chaos created by data augmentation, our method leads to improved feature representations and performance on downstream tasks.
<p align="center">
  <img src="figs/framework.jpg" width="800">
</p>


Contents
----------

* [Datasets](#datasets)
* [Running](#Running)
* [Citation](#citation)

Datasets
----------
1. Datasets
- `Activity recognition`  [UCIHAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), [HHAR](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition), [USC](https://dl.acm.org/doi/pdf/10.1145/2370216.2370438).
- `Heart rate prediction`  [IEEE SPC12 and IEEE SPC22](https://signalprocessingsociety.org/community-involvement/ieee-signal-processing-cup-2015), [DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia).
- `Cardiovascular disease (CVD) classification`  [CPSC2018](http://2018.icbeb.org/Challenge.html), [Chapman](https://digitalcommons.chapman.edu/scs_articles/653/).
2. After downloading the raw data, they should be processed with the corresponding [scripts](https://github.com/eth-siplab/Finding_Order_in_Chaos/tree/main/raw_data_process), if there is any.

Running
----------
The command to run the whole process:
```
python runner_function.py --framework 'simclr' --backbone 'DCL' --dataset 'ucihar' --aug1 'na' --aug2 'resample' --n_epoch 120 --batch_size 256 --lr 3e-3 --lr_cls 0.03 --cuda 0 --cases 'subject_large' --VAE --mean 0.9 --std 0.1
```

If VAE models are not trained in the corresponding folders, e.g., 'ucihar0' for domain 0 of ucihar dataset, the VAE training will start first.

Citation
----------
If you find our paper or codes useful, please cite our work:

 
