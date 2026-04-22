# Comparative analysis of deep learning model outputs using different fine-tuning methods
### Spring 2026 
### Applying Deep Learning to Earth Observation 

## Project Description: 
We are researching the effect of different fine-tuning methods on a pre-trained model called Fieldmapper. Fieldmapper is a toolkit that uses semantic segmentation on cropland using images from satellites. The data set we will be using is called mapping-africa and contains cropland images from Africa. 

## Project Goals
Our goal is to compare how different fine-tuning strategies impact both model performance and the features the model learns. To do this, we will visualize feature representations using dimensionality reduction techniques such as t-SNE, UMAP, and PCA, and analyze how structured the learned features are under each approach. This project aims to better understand how transfer learning strategies affect segmentation performance in Earth observation tasks such as cropland analysis. 

## Methodology:
The fine-tuning strategies we plan to use to test the data are training only the decoder, freezing the first N decoder blocks, progressively unfreezing the encoder during training, and a baseline where the entire model is fine-tuned. All models will be trained under the same conditions to ensure a fair comparison.

We will evaluate performance using metrics such as mean Intersection over Union (mIoU), pixel accuracy, and training/validation loss curves. 

## Expected results
We hypothesize that partial and progressive fine-tuning will outperform a fully frozen encoder because the model can adjust to the specific patterns in satellite images.

## Important Linux commands: 

### To build image: 

```
docker build -t fieldmapper .
```

### To run container (mount 1 drive):

```
docker run -it \
  -v ~/deeplearning:/home/data \
  -v /Volumes/SanDisk/deeplearning_finalproject:/home/models \
  image-name
```

### To run container (mount multiple drives):

```
docker run -it \
  -p 8888:8888 \
  -v ~/deeplearning/fieldmapper:/home/fieldmapper \
  -v ~/deeplearning/transductivefeaturemaps:/home/transductivefeaturemaps \
  -v ~/deeplearning/workdir:/home/workdir \
  -v /Volumes/SanDisk/deeplearning_finalproject:/home/models \
  image-name
```

### To train model:

```
python run_it.py --config ./config/train_finetune.yaml --do-train
```
