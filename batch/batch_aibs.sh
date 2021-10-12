#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=48GB
#SBATCH --array=0-3
#SBATCH --time=5:00:00
#SBATCH -o /network/scratch/g/gillonco/aibs_%A-%a.out 

# 1. load modules
module load anaconda/3
source $CONDA_ACTIVATE

conda activate ssl
python -V

SEED=$((200+SLURM_ARRAY_TASK_ID)) # increment seed for each array task

# check exported variables
if [[ $LFADS == 1 ]]; then
    MODEL=lfads
else
    MODEL=valpaca
fi

if [[ $SUB == 1 ]]; then
    HYPERPARS=$MODEL"_sub.yaml"
    MODEL_DESC=dcen128_dcon64_dgen128_dgla64_dula1_fact32_gene200_ocon128_oenc128_olat128_hp-seed$SEED
    sub_str=" (with submitted hyperparameters)"
else
    HYPERPARS=$MODEL".yaml"
    MODEL_DESC=dcen64_dcon64_dgen64_dgla64_dula1_fact32_gene200_ocon32_oenc32_olat128_hp-seed$SEED
fi

SCRATCH=/network/scratch/g/gillonco
RESULTS_DIR=$SCRATCH/valpaca/results
MODEL_DIR=$RESULTS_DIR/AIBS_m1s1_soma_ou_t0.3_s2.0_f_z_n/$MODEL/$MODEL_DESC

cp $SCRATCH/valpaca/AIBS_m1s1_soma_ou_t0.3_s2.0_f_z_n $SLURM_TMPDIR
DATA_PATH=$SLURM_TMPDIR/AIBS_m1s1_soma_ou_t0.3_s2.0_f_z_n

EXIT=0

# Train model
echo -e "\nTraining $MODEL model$sub_str..."
python train_model.py --model $MODEL --data_suffix fluor --data_path $DATA_PATH --config hyperparameters/AIBS/$HYPERPARS --output_dir $RESULTS_DIR --batch_size 100 --max_epochs 1000 --seed $SEED
code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi

# Move model to prevent future overwriting
NEW_MODEL_DIR=$MODEL_DIR"_job"$SLURM_ARRAY_JOB_ID
echo -e "\nTo prevent future overwriting, moving model from \n$MODEL_DIR \nto \n$NEW_MODEL_DIR..."
mv $MODEL_DIR $NEW_MODEL_DIR

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi 
if [ "$EXIT" -ne 0 ]; then exit $EXIT; fi # exit, if failed

# Infer latent
echo -e "\nInferring latents..."
python infer_latent.py --model_dir $NEW_MODEL_DIR --data_suffix fluor --data_path $DATA_PATH
code="$?"
if [ "$code" -gt "$EXIT" ]; then exit "$code"; fi # exit, if failed


# Run PCA/LogReg
echo -e "\nProducing PCA plots and running logistic regressions..." # most time consuming step...
python analysis/allen_analysis.py --model_dir $NEW_MODEL_DIR --data_path $DATA_PATH --num_runs 50 --projections
code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # exit, if failed


if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi

