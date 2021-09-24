Nodes=2
GPUs=4

srun -N $Nodes --ntasks-per-node=$GPUs --gres=gpu:$GPUs run.sh