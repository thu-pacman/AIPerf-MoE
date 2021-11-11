AIPerf-MoE
===

AIPerf-MoE is a benchmark of AIPerf for systems to train large AI models.
It aims at measuring both computation and communication power of a system.

## Benchmarking guide

All command lines in this document is executed in the root directory of this repository.

### Install prerequisites

This benchmark is based on Megatron-LM's codebase. 
[PyTorch](https://github.com/pytorch/pytorch) and [FastMoE](https://github.com/laekov/fastmoe) are required.
You can install dependents using pypi.

```bash
pip install --user -r requirements.txt
```

### Prepare the data

AIPerf-MoE is using `enwik8` dataset for training.
Run the following command to download a pre-precessed dataset and place it in `data` directory.

```bash
mkdir -p data && curl https://pacman.cs.tsinghua.edu.cn/~laekov/moebench-data.tgz | tar -xz -C data
```

### Configuration

A configuration file is required by AIPerf-MoE, namely `config.yaml`.
An example configuation is shown in `config.default.yaml`.
Modification of any line in this configuation file is allowed for better performance.
Note that validation performance is not counted into final performance.
Therefore, to increase reported performance as high as possible, `eval_interval` should be set large enough in a final run.

Other parts of this benchmark is not allowed to be modified unless necessary.
Please report if a submission involves any modification other than `config.yaml`.

### Start testing

AIPerf-MoE is by default launched using SLURM, and uses 
An example script can be seen in `scripts/run.sh`.

The distributed launcher launches `scripts/pretrain_distributed.sh`, which identifies its rank and world size with SLURM.
If other software, e.g. MPI or PBS, is used to launch the benchmark, modify the environment variables accordingly.

### Scoring

Evaluation is performed after training `train_iters` iterations.
**A valid run requires** the validation loss to be no greater than `3`.

MACs in all attention and MLP layers in forward and backward are counted as FLOP, regardless of their precision.
FP16, FP32 and any other FP format are regarded as the same, as long as the run is valid.
Overall FLOP per second since training begins is the only metric for ranking.
