#!/bin/bash
#SBATCH --output ../out/lorenz_%A.out # Write out
#SBATCH --error ../out/lorenz_%A.err  # Write error
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --job-name=redo_hill
#SBATCH --mem=6GB
#SBATCH --time=6:00:00

module purge
module load anaconda/3
module load pytorch/1.7
source $HOME/valpaca_env/bin/activate

exit_script(){
cp -r $SLURM_TMPDIR/ $PROJECTDIR
}

terminator(){
echo 'job killed'
}

echo $SEED

PROJECTDIR=$HOME/valpaca
HPDIR=$PROJECTDIR/hyperparameters/lorenz
OUTDIR=$PROJECTDIR/models

STATE=seed$SEED'_sys'$DT_SYS'_cal'$DT_CAL'_sig'$SIGMA'_base'$RATE
BASEDATAPATH=$PROJECTDIR/synth_data/lorenz_$STATE

HILL=fluor_hillar1

OU=ou_t0.3_s$OASIS_S
HILLDATAPATH=$BASEDATAPATH'_'$HILL'_'$OU'_n'

python $PROJECTDIR/train_model.py -m valpaca -d $HILLDATAPATH -p $HPDIR/valpaca.yaml -o $OUTDIR --batch_size 40 --max_epochs 2000 --data_suffix $HILL -r

VALPACA_MODEL_DESC=dcen0_dcon0_dgen64_dgla64_dula0_fact3_gene64_ocon32_oenc32_olat64_hp-/
STATE_HILL_OU=$STATE'_'$HILL'_'$OU'_n'

python $PROJECTDIR/infer_latent.py -m $OUTDIR/lorenz_$STATE_HILL_OU/valpaca/$VALPACA_MODEL_DESC -d $HILLDATAPATH --data_suffix $HILL