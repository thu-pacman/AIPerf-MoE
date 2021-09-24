#!/bin/bash
echo loading...
. /opt/spack/share/spack/setup-env.sh
spack load python@3.7.4
spack load openmpi@4.0.5%gcc@8
spack load --first nccl@2.7.8
spack load cuda@11.1
spack load --first py-setuptools@50.3.2
echo finished!

bash ./pretrain_distributed.sh