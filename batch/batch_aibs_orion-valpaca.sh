#!/bin/bash
#SBATCH --array=1-60%10
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=/network/tmp1/bakhtias/exp.%A.%a.out
#SBATCH --error=/network/tmp1/bakhtias/exp.%A.%a.err
#SBATCH --job-name=valpaca_orion_0


module load anaconda/3
source $CONDA_ACTIVATE

conda activate cenv-LFADS-nop
python -V

SCRATCH=/network/tmp1/bakhtias/
RESULTS_DIR=$SCRATCH/Results/valpaca


# cp data from local to compute node
cp $SCRATCH/valpaca/AIBS_m1s1_soma_ou_t0.3_s2.0_f_z_n $SLURM_TMPDIR

export PYTHONUNBUFFERED=1 # do not buffer output to sbatch files

orion hunt -n valpaca_orion_0 train_model.py --config hyperparameters/AIBS/valpaca.yaml --model valpaca --data_suffix fluor --data_path $SLURM_TMPDIR/AIBS_m1s1_soma_ou_t0.3_s2.0_f_z_n  --output_dir $RESULTS_DIR --batch_size 100 --max_epochs 1000 --seed 200