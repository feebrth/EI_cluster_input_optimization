#!/bin/bash
#SBATCH --job-name=study
#SBATCH --output=logs/study%A_%a.out
#SBATCH --error=logs/study_job_%A_%a.err
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --nodelist=agmn-srv-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00

# Load necessary modules
module load python3/anaconda3-2024.06-py3.12

# Activate conda environment
conda activate Nest3


# Run the Python script
python FF_Sim.py

#srun --partition=all --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=3G --time=05:00:00 --nodelist=agmn-srv-4 --pty bash
