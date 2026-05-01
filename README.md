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


## How to filter the images to be only ZM
## Step 1: create your working folders: 
mkdir -p /opt/app-root/src/transductivefeaturemaps/workdir/frozenlayerfinetune/v1
mkdir -p /opt/app-root/src/transductivefeaturemaps/workdir/frozenlayerfinetune/v1/logs
mkdir -p /opt/app-root/src/transductivefeaturemaps/workdir/frozenlayerfinetune/v1/checkpoints

## Step 2: Go into workdir 
cd /opt/app-root/src/transductivefeaturemaps/workdir

## Step 3: Copy the original CSV:
find /opt/app-root/src -name "croplands3-2.1.1_finetune_all_samples.csv" 
(take the path it outputs and run): 
cp <FOUND_PATH> /opt/app-root/src/transductivefeaturemaps/workdir/

## Step 4: Create the ZM-only filtered CSV:
python - <<'PY'
import pandas as pd

infile = "/opt/app-root/src/transductivefeaturemaps/workdir/croplands3-2.1.1_finetune_all_samples.csv"
outfile = "/opt/app-root/src/transductivefeaturemaps/workdir/frozenlayerfinetune/v1/croplands3-2.1.1_finetune_ZM_only.csv"

df = pd.read_csv(infile)

df_zm = df[df["cntry"].astype(str).str.lower() == "zm"].copy()

print("Original rows:", len(df))
print("ZM-only rows:", len(df_zm))
print(df_zm.head())

df_zm.to_csv(outfile, index=False)
print("Saved:", outfile)
PY

## Step 5: Check using:
ls /opt/app-root/src/transductivefeaturemaps/workdir/frozenlayerfinetune/v1
Should see something like: 
croplands3-2.1.1_finetune_ZM_only.csv
logs
checkpoints


