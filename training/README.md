# Multi-Modal DR Training with OpenCLIP
We provide release code and a patch to
[OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main/src/open_clip) 
for training models on DR datasets (DataComp, DFN, or custom datasets generated 
with [MobileCLIP-DR](https://github.com/apple/ml-mobileclip-dr).

In additional to full offline training, we provide support for online knowledge 
distillation from an ensemble of CLIP teachers.

## Data
### DFNDR
Not yet available.

### DataCompDR
Our reinforcements to DataComp are available on HuggingFace.
- [DataCompDR-12M](https://huggingface.co/datasets/apple/DataCompDR-12M)
- [DataCompDR-12M-BFloat16](https://huggingface.co/datasets/apple/DataCompDR-12M-bf16)
- [DataCompDR-1B](https://huggingface.co/datasets/apple/DataCompDR-1B)

Our data does not include the original images and captions. For DataCompDR-12M, 
there is a corresponding 
[DataComp-12M](https://huggingface.co/datasets/mlfoundations/DataComp-12M) with 
original captions. One needs to download both datasets, then run the following 
script to join them:
```bash
#!/bin/bash
DATACOMP12M_PATH="./datasets/DataComp-12M/" # Download path of DataComp-12M from HF
DATACOMPDR12M_NOIMG_PATH="./datasets/DataCompDR-12M-noimage/" # Download path of DataCompDR-12M from HF
DATACOMPDR12M_PATH="./datasets/DataCompDR-12M/"
for  i in {00000000..00001023}
do
  mkdir tmp
  tar -xf $DATACOMP12M_PATH/${i}.tar -C tmp
  tar -xf $DATACOMP12M_NOIMG_PATH/${i}.tar -C tmp
  tar -cf $DATACOMPDR12M_PATH/${i}.tar -C tmp *.*
  rm -rf tmp
done
```

The images have to be downloaded separately. See 
[hf_dataset_example.py](../hf_dataset_example.py) for an example of downloading 
a single image.



## Installing dependencies

The OpenCLIP source used by this project is bundled under
`third_party/open_clip/`, so a
second repository clone is not needed. Install the repository directly:
```bash
# Clone MobileCLIP repository
git clone git@github.com:apple/ml-mobileclip.git
cd ml-mobileclip/
uv sync
```

We retain the v1 patch for reproducibility. The following legacy instructions
intentionally use a separate OpenCLIP checkout and are only needed to reproduce
the v1 implementation:
```
# Revert changes for compatibility with older OpenCLIP commit
sed -i 's/open_clip_train/training/g' ../dr/transforms.py
find . --name "*.sh" | xargs sed -i 's/open_clip_train/training/g'

# Clone OpenCLIP repository, apply patch, and install
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
git checkout cf86ee7ec4658845f640858ecd34d0f15588271a  # Wed May 29 21:57:08 2024 +0700
git apply ../open_clip_v1.patch
cp ../configs/ ./ -r
cp ../dr/ ./src/training/ -r
```

## Training

### DFNDR
We provide scripts for training on DFNDR-2B12M and DFNDR-2B.

```bash
bash training/configs/run_dfndr12m.sh  # Train a MobileCILP-B on DFN-2B12M with DR
bash training/configs/run_dfndr2b.sh  # Train a MobileCLIP-B on DFN-2B with DR
```

### DataCompDR

We provide scripts for training on DataCompDR-12M and DataCompDR-1B.

```bash
bash training/configs/run_datacomp12m.sh  # Train a ViT-B/16 on DataComp-12M without DR
bash training/configs/run_datacompdr12m.sh  # Train a ViT-B/16 on DataComp-12M with DR
bash training/configs/run_datacompdr1b.sh  # Train a ViT-B/16 on DataComp-1B with DR
```
