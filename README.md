# MAF: A General Matching and Alignment Framework for Multimodal Named Entity Recognition

Pytorch Implementation of "MAF: A General Matching and Alignment Framework for Multimodal Named Entity Recognition" (WSDM 2022)

The code implementation of this paper refers to [jefferyYu](https://github.com/jefferyYu/UMT).


<img src="https://github.com/xubodhu/MAF/framework.png"  width="897" height="317" />

## Requirements

python>=3.6

torch>=1.7.1

pytorch-crf>=0.7.2



## Configuration

--negative_rate 2k in Cross-Modal Matching Module

--lamb Parameters between the main loss function and other task loss functions

--temp Temperature parameter in Cross-Modal Alignment Module

--temp_lamb 𝜆c in Cross-Modal Alignment Module



## Datasets

See [jefferyYu](https://github.com/jefferyYu/UMT).



## Usage

- Download the data from [jefferyYu](https://github.com/jefferyYu/UMT).

- Follow the link provided by [jefferyYu](https://github.com/jefferyYu/UMT) to download the pre-trained ResNet-152 from   to the folder resnet.

```bash

sh run.sh


```