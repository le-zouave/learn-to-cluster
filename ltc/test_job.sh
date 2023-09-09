#!/bin/bash
#SBATCH --account=def-hezaveh
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:00:00

module load StdEnv/2020 gcc/9.3.0 python/3.10 cuda/11.4 faiss/1.7.3 igraph/0.10.2 opencv/4.7

virtualenv --no-download --clear $SLURM_TMPDIR/env && source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -U pip

cd $LTU_PATH/learn-to-cluster/
pip install --no-index -r requirements.txt
python setup.py develop --user

bash $LTU_PATH/learn-to-cluster/ltc/scripts/lgcn/test_lgcn_fashion.sh
