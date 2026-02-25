# STEAD-AI

The AI model behind the [GV20 STEAD Platform](https://www.nature.com/articles/d43747-023-00067-3)
for deep learning on B cell receptor (BCR) repertoires.

## Overview

STEAD-AI is a collection of deep learning models that learn associations between
antibody sequences and patient profiles such as cancer type, gene expression,
and survival outcome. Built on a permutation-invariant max-pooling architecture,
the models analyze variable-size BCR repertoires from high-throughput sequencing
to produce patient-level predictions, enabling applications in anti-cancer target
discovery and immunological profiling.

### Models

- **PhialBCR** -- Core model with amino acid encoding, isotype-aware layers,
  and max-pooling over BCR repertoires
- **PhialBCR_batch** -- PhialBCR with batch normalization
- **PhialBCR_MTL** -- Multi-task learning model capable of jointly learning
  associations between antibody sequences and multiple patient profiles
  (e.g., cancer subtype, anti-cancer targets, survival) in a single training run

For detailed architecture descriptions, training algorithms, and mathematical
formulations, see the [Model Architectures](model_architectures.md) reference.

## Dependencies

- Python >= 3.8
- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- lifelines

```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
  phial_bcr.py        # Core deep learning model classes
  phial_bcr_mtl.py    # Multitask learning extension
  tcga_bcr.py         # Data loading and preprocessing for TCGA BCR data
  tcga_bcr_train.py   # Training script with CLI arguments
```

## Usage

### Training a model

```bash
cd src
python tcga_bcr_train.py \
    --model PhialBCR_MTL \
    --dataset MDS5 \
    --datapath ../data \
    --workpath ../work \
    --max-epoch 50
```

### Running module tests

```bash
cd src
python phial_bcr.py --workpath ../work
python phial_bcr_mtl.py --workpath ../work
```

## Data

The MDS5 dataset is proprietary to GV20 Therapeutics and is not provided in this
repository.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use STEAD-AI in your research, please cite our paper (forthcoming).
