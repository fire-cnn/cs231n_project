#!/usr/bin/bash
#SBATCH --job-name=run_ft_vit
#SBATCH --output=ft_vit_2.out
#SBATCH --error=ft_vit_2.err
#SBATCH --time=01:00:00
#SBATCH -p serc
#SBATCH --nodes 1
#SBATCH --mem 16GB
#SBATCH -c 5
#SBATCH --gpus 1
#SBATCH --mail-user=ihigueme@stanford.edu
#SBATCH --mail-type=ALL

source /scratch/users/ihigueme/cs231n_project/bin/activate

python3 vit_ft.py \
--config_file config_sherlock_vit.yaml \
--dir_project /scratch/users/ihigueme/wandb_dir \
--tags vit_simple,classification,sherlock,new_metrics






























