#!/bin/bash
#SBATCH --job-name=study
#SBATCH --output=logs/study%A_%a.out
#SBATCH --error=logs/study_job_%A_%a.err
#SBATCH --array=0-9
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=04:59:00

# Load necessary modules
module load python3/anaconda3-2024.06-py3.12

# Activate conda environment
conda activate Nest3


# Run the Python script
python Optuna_8_Stimuli.py
