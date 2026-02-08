#!/bin/bash
#SBATCH --job-name=ddtf_gpu     # Name of the job
#SBATCH --output=log_ddtf_freq.out    # Output file name pattern (%A for array master ID, %a for array index)
#SBATCH --error=log_ddtf_freq.err      # Error file name pattern
#SBATCH --mem=64G                        # Memory per task
#SBATCH -N 1
#SBATCH --gres=gpu:1 ## Run on 1 GPU
#SBATCH --ntasks=1                     # Total number of tasks across all nodes
#SBATCH --ntasks-per-node=1              # Each node can handle only one task due to the 96 GB memory requirement
#SBATCH --cpus-per-task=24               # Assuming each node has 48 cores
#SBATCH -p dgx_aic                     # Partition name

# Load necessary modules
module load cuda/11.3
module load python3/anaconda/2023.7

export SPECTRAL_CONNECTIVITY_ENABLE_GPU="true"
# Run the Python script with array task ID as an argument
cd /work/srishyla
python ddtf.py

