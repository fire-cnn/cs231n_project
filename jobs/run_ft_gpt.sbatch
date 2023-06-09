#!/usr/bin/bash
#SBATCH --job-name=run_ft_gpt
#SBATCH --output=ft_gpt.out
#SBATCH --error=ft_gpt.err
#SBATCH --time=01:00:00
#SBATCH -p serc
#SBATCH --nodes 1
#SBATCH --mem 16GB
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH --mail-user=ihigueme@stanford.edu
#SBATCH --mail-type=ALL


source /scratch/users/ihigueme/cs231n_project/bin/activate

python3 gpt_2_ft.py \
--config_file config_sherlock_gpt.yaml \
--dir_project /scratch/users/ihigueme/wandb_dir \
--tags gpt2,classification,sherlock,gpt,f1





























