# FuncGNN
This repository is the corresponding model repository for the paper "FuncGNN: Learning Functional Semantics of Logic Circuits with Graph Neural Networks"

## Environment Setup

- **GPU**: NVIDIA A800 * 1
- **CUDA Version** 11.8.0
- **OS**: Ubuntu 22.04.5 LTS  

## Conda Environment

```
conda create -n funcgnn python=3.8.20
conda activate funcgnn
pip install -r requirements.txt
```

## Dataset preparing

Dataset from: https://github.com/BUPT-GAMMA/PolarGate/releases/download/dataset/PolarGate_processed.zip

```
cd ~/
mkdir AIGDataset
cd AIGDataset
wget https://github.com/BUPT-GAMMA/PolarGate/releases/download/dataset/PolarGate_processed.zip
unzip PolarGate_processed.zip 
```

## Training Configuration

- **Training Split**:  
  - Training: 5%  
  - Validation: 5%  
  - Test: 90%  
- **Batch Size**: 128  
- **Epochs**: Up to 500  
- **Early Stopping**: Stops if no improvement for 100 epochs  

## Model Training

SPP Task

```
python train.py --task prob --device 0 --batch_size 128
```

TTDP Task

```
python train.py --task tt --device 0 --batch_size 128
```

## Cite FuncGNN

If FuncGNN could help your project, please cite our work:

```

```

