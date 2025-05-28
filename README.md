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

## Model Training

```
python train.py --task prob --device 0 --batch_size 128 
```

