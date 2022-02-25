#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48GB
#SBATCH --array=0-7
#SBATCH --time=4:00:00
#SBATCH -o /network/scratch/g/gillonco/allen_%A-%a.out 

# 1. load modules
module load anaconda/3
source activate ssl

python -V

SEED=$((200+SLURM_ARRAY_TASK_ID)) # increment seed for each array task


# ORION HYPERPARAMS

# check exported variables, and predict model directory name
if [[ $LFADS == 1 ]]; then
    SUB=1
    
    MODEL=lfads
    DIREC=lfads_oasis
    DATA_SUFFIX=ospikes

    # params
    cENC=128
    FACT=32
    gENC=128
    GENE=200
    gLAT=64
    uLAT=1

    MODEL_DESC='cenc'$cENC'_cont'$CONT'_fact'$FACT'_genc'$gENC'_gene'$GENE'_glat'$gLAT'_ulat'$uLAT'_hp-seed'$SEED
else
    MODEL=valpaca
    DIREC=$MODEL
    DATA_SUFFIX=fluor

    if [[ $SUB == 1 ]]; then
        DEEP=128
        dCTRL=64
        OBS=128
    else
        DEEP=64
        dCTRL=32
        OBS=128
    fi

    # params
    dCEN=$DEEP
    dCTRL=$dCTRL
    dGEN=$DEEP
    dGLA=64
    duLA=1
    FACT=32
    GEN=200
    oCON=$OBS
    oENC=$OBS
    oLAT=128

    MODEL_DESC='dcen'$dCEN'_dcon'$dCTRL'_dgen'$dGEN'_dgla'$dGLA'_dula'$duLA'_fact'$FACT'_gene'$GEN'_ocon'$oCON'_oenc'$oENC'_olat'$oLAT'_hp-seed'$SEED
fi

if [[ $SUB == 1 ]]; then
    HYPERPARS=$MODEL"_sub.yaml"
    sub_str=" (with submitted hyperparameters)"
else
    HYPERPARS=$MODEL".yaml"
fi

SCRATCH=/network/scratch/g/gillonco
RESULTS_DIR=$SCRATCH/valpaca/results
MODEL_DIR=$RESULTS_DIR/allen_m1s1_soma_ou_t0.3_s2.0_f_z_n/$DIREC/$MODEL_DESC

cp $SCRATCH/valpaca/allen_m1s1_soma_ou_t0.3_s2.0_f_z_n $SLURM_TMPDIR
DATA_PATH=$SLURM_TMPDIR/allen_m1s1_soma_ou_t0.3_s2.0_f_z_n

EXIT=0

# Train model
echo -e "\nTraining $MODEL model$sub_str..."
python train_model.py \
    --restart \
    --model $MODEL \
    --data_suffix $DATA_SUFFIX \
    --data_path $DATA_PATH \
    --config hyperparameters/allen/$HYPERPARS \
    --output_dir $RESULTS_DIR \
    --batch_size 68 \
    --max_epochs 800 \
    --seed $SEED

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi

# Move model to prevent eventual overwriting
NEW_MODEL_DIR=$MODEL_DIR"_job"$SLURM_ARRAY_JOB_ID
echo -e "\nTo prevent eventual overwriting, moving model from \n$MODEL_DIR \nto \n$NEW_MODEL_DIR..."
mv $MODEL_DIR $NEW_MODEL_DIR

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi 
if [ "$EXIT" -ne 0 ]; then exit $EXIT; fi # exit, if failed


# Infer latent
echo -e "\nInferring latents..."
python infer_latent.py \
    --model_dir $NEW_MODEL_DIR \
    --data_suffix $DATA_SUFFIX \
    --data_path $DATA_PATH

code="$?"
if [ "$code" -gt "$EXIT" ]; then exit "$code"; fi # exit, if failed


# Run PCA and decoders
echo -e "\nProducing PCA factor plots and running decoders..." # most time consuming step...
python analysis/allen_analysis.py \
    --model_dir $NEW_MODEL_DIR \
    --data_path $DATA_PATH \
    --run_svm \
    --run_logreg \
    --run_nl_dec \
    --num_runs 20 \
    --projections \
    --seed $SEED

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # exit, if failed


if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi

