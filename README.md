# Diverse Sampling
> Official project of ACMMM2022 accepted paper "[Diverse Human Motion Prediction via Gumbel-Softmax Sampling from an Auxiliary Space]()"

[\[Paper\]]()
[\[Supp\]]()
[\[Poster\]]()
[\[Slides\]]()
[\[Video\]]()


## Authors

1. [Lingwei Dang](https://github.com/Droliven), School of Computer Science and Engineering, South China University of Technology, China, [danglevon@gmail.com](mailto:danglevon@gmail.com)
2. [Yongwei Nie](https://nieyongwei.net), School of Computer Science and Engineering, South China University of Technology, China, [nieyongwei@scut.edu.cn](mailto:nieyongwei@scut.edu.cn)
3. [Chengjiang Long](http://www.chengjianglong.com), Meta Reality Lab, USA, [cjfykx@gmail.com](mailto:cjfykx@gmail.com)
4. [Qing Zhang](http://zhangqing-home.net/), School of Computer Science and Engineering, Sun Yat-sen University, China, [zhangqing.whu.cs@gmail.com](mailto:zhangqing.whu.cs@gmail.com)
5. [Guiqing Li](http://www2.scut.edu.cn/cs/2017/0629/c22284a328097/page.htm), School of Computer Science and Engineering, South China University of Technology, China, [ligq@scut.edu.cn](mailto:ligq@scut.edu.cn)

## Abstract
###### &nbsp;&nbsp;&nbsp; Diverse human motion prediction aims at predicting multiple possible future pose sequences from a sequence of observed poses. Previous approaches usually employ deep generative networks to model the conditional distribution of data, and then randomly sample outcomes from the distribution. While different results can be obtained, they are usually the most likely ones which are not diverse enough. Recent work explicitly learns multiple modes of the conditional distribution via a deterministic network, which however can only cover a fixed number of modes within a limited range. In this paper, we propose a novel sampling strategy for sampling very diverse results from an imbalanced multimodal distribution learned by a deep generative model. Our method works by generating an auxiliary space and smartly making randomly sampling from the auxiliary space equivalent to the diverse sampling from the target distribution. We propose a simple yet effective network architecture that implements this novel sampling strategy, which incorporates a Gumbel-Softmax coefficient matrix sampling method and an aggressive diversity promoting hinge loss function. Extensive experiments demonstrate that our method significantly improves both the diversity and accuracy of the samplings compared with previous state-of-the-art sampling approaches.

## Overview

<a href="./assets/7627-poster.pdf">
  <img src="./assets/7627-poster.png" />
</a>



## Dependencies

* Pytorch 1.10.0+cu113
* Python 3.9.7
* Nvidia RTX 3090

[//]: # (## Get the data)

[//]: # ([Human3.6m]&#40;http://vision.imar.ro/human3.6m/description.php&#41; in exponential map can be downloaded from [here]&#40;http://www.cs.stanford.edu/people/ashesh/h3.6m.zip&#41;.)

[//]: # ()
[//]: # ([HumanEva-I]&#40;http://mocap.cs.cmu.edu/&#41; was obtained from the [repo]&#40;https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics&#41; of ConvSeq2Seq paper.)

[//]: # ()
[//]: # (## About datasets)

[//]: # ()
[//]: # (Human3.6M dataset)

[//]: # ()
[//]: # (+ A pose in h3.6m has 32 joints, from which we choose 22, and build the multi-scale by 22 -> 12 -> 7 -> 4 dividing manner.)

[//]: # (+ We use S5 / S11 as test / valid dataset, and the rest as train dataset, testing is done on the 15 actions separately, on each we use all data instead of the randomly selected 8 samples.)

[//]: # (+ Some joints of the origin 32 have the same position)

[//]: # (+ The input / output length is 10 / 25)

[//]: # ()
[//]: # (HumanEva-I dataset)

[//]: # ()
[//]: # (+ A pose in cmu has 38 joints, from which we choose 25, and build the multi-scale by 25 -> 12 -> 7 -> 4 dividing manner.)

[//]: # (+ CMU does not have valid dataset, testing is done on the 8 actions separately, on each we use all data instead of the random selected 8 samples.)

[//]: # (+ Some joints of the origin 38 have the same position)

[//]: # (+ The input / output length is 10 / 25)

[//]: # ()
[//]: # (## Train)

[//]: # ()
[//]: # (+ train on Human3.6M:)

[//]: # ()
[//]: # (  `python main.py --exp_name=h36m --is_train=1 --output_n=25 --dct_n=35 --test_manner=all`)

[//]: # ()
[//]: # (+ train on CMU Mocap:)

[//]: # ()
[//]: # (  `python main.py --exp_name=cmu --is_train=1 --output_n=25 --dct_n=35 --test_manner=all`)

[//]: # ()
[//]: # ()
[//]: # (## Evaluate and visualize results)

[//]: # ()
[//]: # (+ evaluate on Human3.6M:)

[//]: # ()
[//]: # (  `python main.py --exp_name=h36m --is_load=1 --model_path=ckpt/pretrained/h36m_in10out25dctn35_best_err57.9256.pth --output_n=25 --dct_n=35 --test_manner=all`)

[//]: # ()
[//]: # (+ evaluate on CMU Mocap: )

[//]: # (  )
[//]: # (  `python main.py --exp_name=cmu --is_load=1 --model_path=ckpt/pretrained/cmu_in10out25dctn35_best_err37.2310.pth --output_n=25 --dct_n=35 --test_manner=all`)

[//]: # ()
[//]: # (## Results)

[//]: # ()
[//]: # (H3.6M-10/25/35-all | 80 | 160 | 320 | 400 | 560 | 1000 | -)

[//]: # (:----: | :----: | :----: | :----: | :----: | :----: | :----: | :----:)

[//]: # (walking | 12.16 | 22.65 | 38.65 | 45.24 | 52.72 | 63.05 | -)

[//]: # ()
[//]: # ()
[//]: # (****)

[//]: # ()
[//]: # (CMU-10/25/35-all | 80 | 160 | 320 | 400 | 560 | 1000 | -)

[//]: # (:----: | :----: | :----: | :----: | :----: | :----: | :----: | :----:)

[//]: # (basketball | 10.24 | 18.64 | 36.94 | 45.96 | 61.12 | 86.24 | -)


  
## Citation

If you use our code, please cite our work

```
@InProceedings{Dang_2022_acmmm,
    author    = {Dang, Lingwei and Nie, Yongwei and Long, Chengjiang and Zhang, Qing and Li, Guiqing},
    title     = {Diverse Human Motion Prediction via Gumbel-Softmax Sampling from an Auxiliary Space},
    booktitle = {Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)},
    month     = {October},
    year      = {2022},
}
```

## Acknowledgments

We follow the code framework of our previous work [MSR-GCN](https://github.com/Droliven/MSRGCN), and some code was adapted from [DLow](https://github.com/Khrylx/DLow) by [Ye Yuan](https://github.com/Khrylx), and [GSPS](https://github.com/wei-mao-2019/gsps) by [Wei Mao](https://github.com/wei-mao-2019). 

## Licence
MIT
