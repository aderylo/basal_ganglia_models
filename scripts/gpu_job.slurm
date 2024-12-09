#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --partition=plgrid-gpu-a100
#SBATCH --job-name=rl_training_job-"$@"
#SBATCH --output=jobs/%j-rl_training_job_output.txt
#SBATCH --error=jobs/%j-rl_training_job_error.txt
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1

cd $PLG_GROUPS_STORAGE/plggcompneuro/snn/basal_ganglia_models

# Load necessary modules
module load CUDA/12.4.0
module load GCCcore/12.3.0
module load Python/3.11.3
module load Mesa/23.1.4

# Check the enviorement
source .venv/bin/activate
pip freeze

# Check if you have indeed a gpu
nvidia-smi

 
# Run your Python script
ls
echo $j
echo "$@"
python $@ >> jobs/\$SLURM_JOB_ID-script-out.txt 2>>jobs/\$SLURM_JOB_ID-script-err.txt
EOT
