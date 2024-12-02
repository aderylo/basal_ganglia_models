#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --partition=plgrid-gpu-a100
#SBATCH --job-name=rl_training_job-"$@"
#SBATCH --output=jobs/%j-rl_training_job_output.txt
#SBATCH --error=jobs/%j-rl_training_job_error.txt
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1

cd $PLG_GROUPS_STORAGE/plggcompneuro/snn/basal_ganglia_models

# Load necessary modules
module load cuda/10.2
module load GCCcore/12.3.0
module load Mesa/23.1.4
module load Miniconda3/23.3.1-0

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate conda activate /net/pr2/projects/plgrid/plggcompneuro/snn/basal_ganglia_models.git/.conda/ppo 

# Check if you have indeed a gpu
nvidia-smi

 
# Run your Python script
ls
echo $j
echo "$@"
python3.11 $@ >> jobs/\$SLURM_JOB_ID-script-out.txt 2>>jobs/\$SLURM_JOB_ID-script-err.txt
EOT
