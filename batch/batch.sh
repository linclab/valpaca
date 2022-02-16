#!/bin/bash
#SBATCH --cpus-per-task=8                     # Ask for 8 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=48GB                            # Ask for 48 GB of RAM
#SBATCH --time=3:00:00                        # The job will run for 3 hours
#SBATCH -o /network/tmp1/bakhtias/slurm-%j.out  # Write the log on tmp1

# 1. load modules
module load anaconda/3
source $CONDA_ACTIVATE

# 1. Load your environment
conda activate cenv-LFADS-nop
python -V
# 2. Copy your dataset on the compute node
# cp ./synth_data/lorenz_750 $SLURM_TMPDIR
cp ./allen_m1s1_soma_ou_t0.3_s2.0_f_z_n $SLURM_TMPDIR
#unzip -q $SLURM_TMPDIR/UCF101.zip -d $SLURM_TMPDIR

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
#cd ./process_data/src/
#python write_csv.py

#cd ../../dpc

# python train_model.py --model 'conv3d_lfads' --data_path $SLURM_TMPDIR/lorenz_750 --hyperparameter_path /home/mila/b/bakhtias/Project-Codes/hierarchical_lfads/hyperparameters/lorenz/conv3d_lfads.yaml --output_dir /network/tmp1/bakhtias/Results/LFADS/ --batch_size 60 --max_epochs 500
# python -u train_model.py --model 'conv3d_lfads' --data_path $SLURM_TMPDIR/lorenz_750 --hyperparameter_path /home/mila/b/bakhtias/Project-Codes/hierarchical_lfads/hyperparameters/lorenz/conv3d_lfads.yaml --output_dir /network/tmp1/bakhtias/Results/LFADS/ --batch_size 40 --max_epochs 2000

python train_model.py --model 'lfads-gaussian' --data_suffix fluor --data_path $SLURM_TMPDIR/allen_m1s1_soma_ou_t0.3_s2.0_f_z_n --config /home/mila/b/bakhtias/Project-Codes/calfads/hyperparameters/lorenz/lfads-gaussian.yaml --output_dir /network/tmp1/bakhtias/Results/LFADS/ --batch_size 100 --max_epochs 1000

# python run_conv3d_lfads.py -d $SLURM_TMPDIR/lorenz_750 -p /network/home/bakhtias/Project-Codes/hierarchical_lfads/hyperparameters/lorenz/conv3d_lfads.yaml --batch_size=60 --output_dir=/network/tmp1/bakhtias/Results/LFADS --max_epochs=10

# 4. Copy whatever you want to save on $SCRATCH

cp -r $SLURM_TMPDIR /network/tmp1/bakhtias/Results/LFADS


