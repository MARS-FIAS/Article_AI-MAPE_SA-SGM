#!/bin/bash
#SBATCH --job-name=Shallow_Grid_1_N_Link_Inference_Stage
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=4400
#SBATCH --extra-node-info=2:20:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL

#SBATCH --array=0

#SBATCH --output=%x_%A_%a.out

#SBATCH --begin=now+0hour

task_index=$SLURM_ARRAY_TASK_ID
task="./Script_Bank/Prime/${SLURM_JOB_NAME}.py" # Careful! The user must provide the path to its own HPC directory!
python $task $task_index
