# Environment

```
pip install -r requirements.txt
```



# Datasets

This project uses cifar10, NUSWIDE and Imagenet datasets. If you need to download, you can access  

[test.ipynb](https://github.com/q878787/DAMH/blob/main/test.ipynb)  to download from the kaggle address. In addition，cifar10 is automatically downloaded by pytorch。

In addition, Deepfashion from Kaggle(https://www.kaggle.com/datasets/hserdaraltan/deepfashion-inshop-clothes-retrieval)

Note: please put the datasets in the dataset folder



# Run

```
python DAMH-easy-version.py
```



# precision-recall curve

1.The experimental results of DAMH(48bit) and comparison methods on the Cifar10 dataset under three evaluation metrics

![cifar](https://raw.githubusercontent.com/q878787/images/main/markdownImg/202207312158496.png)

2.The experimental results of DAMH(48bit) and comparison methods on the NUSWIDE dataset under three evaluation metrics

![nuswide](https://raw.githubusercontent.com/q878787/images/main/markdownImg/202207312159256.png)

3.The experimental results of DAMH(48bit) and comparison methods on the Imagenet dataset under three evaluation metrics

![imagenet](https://raw.githubusercontent.com/q878787/images/main/markdownImg/202207312159543.png)



# Mean Average Precision,48 bits

| Algorithms | dataset                | map   |
| ---------- | ---------------------- | ----- |
| DAMH       | cifar10                | 0.807 |
|            | nus_wide_21            | 0.822 |
|            | imagenet               | 0.657 |
|            | deepfashion(RestNet18) | 0.551 |
| DCH        | cifar10                | 0.793 |
|            | nus_wide_21            | 0.71  |
|            | imagenet               | 0.664 |
|            | deepfashion(RestNet18) | 0.513 |
| DTSH       | cifar10                | 0.773 |
|            | nus_wide_21            | 0.82  |
|            | imagenet               | 0.644 |
|            | deepfashion(RestNet18) | 0.498 |
|            |                        |       |
