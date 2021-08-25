#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=48GB
#SBATCH --time=1:00:00
#SBATCH -o /miniscratch/gillonco/allen-valpaca-%j.out 

# 1. load modules
module load anaconda/3
source $CONDA_ACTIVATE

conda activate valpaca-env
python -V

SEED=200

SCRATCH=/miniscratch/$USER
RESULTS_DIR=$SCRATCH/valpaca/results
VALPACA_MODEL_DESC=dcen128_dcon64_dgen128_dgla64_dula1_fact32_gene200_ocon128_oenc128_olat128_hp-seed$SEED
MODEL_DIR=$RESULTS_DIR/AIBS_m1s1_soma_ou_t0.3_s2.0_f_z_n/valpaca/$VALPACA_MODEL_DESC

cp $SCRATCH/valpaca/AIBS_m1s1_soma_ou_t0.3_s2.0_f_z_n $SLURM_TMPDIR

EXIT=0

python train_model.py --model valpaca --data_suffix fluor --data_path $SCRATCH/valpaca/AIBS_m1s1_soma_ou_t0.3_s2.0_f_z_n --hyperparameter_path hyperparameters/AIBS/valpaca.yaml --output_dir $RESULTS_DIR --batch_size 100 --max_epochs 1000 --seed $SEED
code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi


python infer_latent.py --model_dir $MODEL_DIR --data_suffix fluor --data_path $SCRATCH/valpaca/AIBS_m1s1_soma_ou_t0.3_s2.0_f_z_n 
code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi


if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi
