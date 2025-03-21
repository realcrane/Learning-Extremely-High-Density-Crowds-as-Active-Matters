## [Learning Extremely High Density Crowds as Active Matters (CVPR 2025)](https://arxiv.org/abs/2503.12168)
![Paper Image](./framework.jpg)

Video-based high-density crowd analysis and prediction has been a long-standing topic in computer vision. It is notoriously difficult due to, but not limited to, the lack of high-quality data and complex crowd dynamics. Consequently, it has been relatively under studied. In this paper, we propose a new approach that aims to learn from in-the-wild videos, often with low quality where it is difficult to track individuals or count heads. The key novelty is a new physics prior to model crowd dynamics. We model high-density crowds as active matter, a continumm with active particles subject to stochastic forces, named `crowd material'. Our physics model is combined with neural networks, resulting in a neural stochastic differential equation system which can mimic the complex crowd dynamics. Due to the lack of similar research, we adapt a range of existing methods which are close to ours for comparison. Through exhaustive evaluation, we show our model outperforms existing methods in analyzing and forecasting extremely high-density crowds. Furthermore, since our model is a continuous-time physics model, it can be used for simulation and analysis, providing strong interpretability. This is categorically different from most deep learning methods, which are discrete-time models and black-boxes.

## Getting Started
### Dependencies
Below is the key environment under which the code was developed, not necessarily the minimal requirements:

1. Python 3.8.18
2. pytorch 1.8.2

And other libraries such as numpy.

### Prepare data
The preprocessed data includes Drill, Marathon and Hellfest, you can download the data from [here](http://pan.csu.edu.cn:80/link/58A31DB82E0F4AE4493F4259AE381854).

### Authors
Feixiang He, Jiangbei Yue, Jialin Zhu, Armin Seyfried, Dan Casas, Julien Pettré, He Wang

Feixiang He, drfxhe@gmail.com, [Homepage](https://feixianghe.github.io/)

He Wang, he_wang@ucl.ac.uk, [Personal website](https://drhewang.com)

### Contact
If you have any questions, please contact me: Feixiang He (drfxhe1992@gmail.com)

### Citation (Bibtex)
Please cite our paper if you find it useful:

