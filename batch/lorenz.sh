#!/bin/bash
conda activate pytorch17

PROJECTDIR=$WORK/valpaca_v4 
HPDIR=$PROJECTDIR/hyperparameters/lorenz
OUTDIR=$PROJECTDIR/models

python $PROJECTDIR/generate_synthetic_data.py -o $PROJECTDIR/synth_data -s $SEED -p $PROJECTDIR/synth_data/lorenz_params.yaml --dt_sys $DT_SYS --dt_spike $DT_CAL --sigma $SIGMA --rate_scale $RATE

STATE=seed$SEED'_sys'$DT_SYS'_cal'$DT_CAL'_sig'$SIGMA'_base'$RATE
BASEDATAPATH=$PROJECTDIR/synth_data/lorenz_$STATE



AR1=fluor_ar1
HILL=fluor_hillar1

python $PROJECTDIR/preprocessing_oasis.py -d $BASEDATAPATH -t 0.3 -s $OASIS_S --data_suffix $AR1
python $PROJECTDIR/preprocessing_oasis.py -d $BASEDATAPATH -t 0.3 -s $OASIS_S --data_suffix $HILL -n

OU=ou_t0.3_s$OASIS_S
AR1DATAPATH=$BASEDATAPATH'_'$AR1'_'$OU
HILLDATAPATH=$BASEDATAPATH'_'$HILL'_'$OU'_n'

python $PROJECTDIR/train_model.py -m lfads -d $AR1DATAPATH -p $HPDIR/lfads.yaml -o $OUTDIR --batch_size 40 --max_epochs 2000 --data_suffix spikes

python $PROJECTDIR/train_model.py -m lfads -d $AR1DATAPATH -p $HPDIR/lfads.yaml -o $OUTDIR --batch_size 40 --max_epochs 2000 --data_suffix $AR1'_'ospikes
python $PROJECTDIR/train_model.py -m lfads-gaussian -d $AR1DATAPATH -p $HPDIR/lfads-gaussian.yaml -o $OUTDIR --batch_size 40 --max_epochs 2000 --data_suffix $AR1
 python $PROJECTDIR/train_model.py -m valpaca -d $AR1DATAPATH -p $HPDIR/valpaca.yaml -o $OUTDIR --batch_size 40 --max_epochs 2000 --data_suffix $AR1

python $PROJECTDIR/train_model.py -m lfads -d $HILLDATAPATH -p $HPDIR/lfads.yaml -o $OUTDIR --batch_size 40 --max_epochs 2000 --data_suffix $HILL'_'ospikes
python $PROJECTDIR/train_model.py -m lfads-gaussian -d $HILLDATAPATH -p $HPDIR/lfads-gaussian.yaml -o $OUTDIR --batch_size 40 --max_epochs 2000 --data_suffix $HILL
python $PROJECTDIR/train_model.py -m valpaca -d $HILLDATAPATH -p $HPDIR/valpaca.yaml -o $OUTDIR --batch_size 40 --max_epochs 2000 --data_suffix $HILL

LFADS_MODEL_DESC=cenc0_cont0_fact3_genc64_gene64_glat64_ulat0_hp-/
VALPACA_MODEL_DESC=dcen0_dcon0_dgen64_dgla64_dula0_fact3_gene64_ocon32_oenc32_olat64_hp-/
STATE_AR1_OU=$STATE'_'$AR1'_'$OU
STATE_HILL_OU=$STATE'_'$HILL'_'$OU'_n'

python $PROJECTDIR/infer_latent.py -m $OUTDIR/lorenz_$STATE_AR1_OU/lfads/$LFADS_MODEL_DESC -d $AR1DATAPATH --data_suffix spikes
python $PROJECTDIR/infer_latent.py -m $OUTDIR/lorenz_$STATE_AR1_OU/lfads-gaussian/$LFADS_MODEL_DESC -d $AR1DATAPATH --data_suffix $AR1
python $PROJECTDIR/infer_latent.py -m $OUTDIR/lorenz_$STATE_AR1_OU/lfads_oasis/$LFADS_MODEL_DESC -d $AR1DATAPATH --data_suffix $AR1'_'ospikes
python $PROJECTDIR/infer_latent.py -m $OUTDIR/lorenz_$STATE_AR1_OU/valpaca/$VALPACA_MODEL_DESC -d $AR1DATAPATH --data_suffix $AR1
python $PROJECTDIR/infer_latent.py -m $OUTDIR/lorenz_$STATE_HILL_OU/lfads-gaussian/$LFADS_MODEL_DESC -d $HILLDATAPATH --data_suffix $HILL
python $PROJECTDIR/infer_latent.py -m $OUTDIR/lorenz_$STATE_HILL_OU/lfads_oasis/$LFADS_MODEL_DESC -d $HILLDATAPATH --data_suffix $HILL'_'ospikes
python $PROJECTDIR/infer_latent.py -m $OUTDIR/lorenz_$STATE_HILL_OU/valpaca/$VALPACA_MODEL_DESC -d $HILLDATAPATH --data_suffix $HILL