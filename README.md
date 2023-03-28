# VaLPACa

## 1. Description
This repository contains code to run the VaLPACa algorithm (Variational Ladders for Parallel Autoencoding of Calcium imaging data), as published in [Prince _et al._, 2021, _biorXiv_](https://www.biorxiv.org/content/10.1101/2021.03.05.434105v1).   

## 2. Installation
The code is written in `Python 3`.  
Requirements for running the scripts in this repository are listed in `requirements.txt`.

## 3. Use
To train VaLPACa on a dataset, run `train_model.py`.  
To extract latents from data, using a trained model, run `infer_latent.py`.  

## 4. Modules and co.
To train VaLPACa on a dataset, run `train_model.py`.  
To extract latents from data, using a trained model, run `infer_latent.py`.  
* `analysis`: latent analysis modules  
* `batch`: example slurm scripts  
* `data`: data generation and processing modules  
* `hyperparameters`: yaml files defining hyperparameter values for different models and datasets  
* `models`: modules defining models, and objective functions  
* `utils`: processing, training, and plotting utilities  

## 5. Example notebook

Follow the link below for an example of how to run VaLPACa on new data.  
[![Run the Google Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/linclab/valpaca/blob/main/run_valpaca_example.ipynb)
