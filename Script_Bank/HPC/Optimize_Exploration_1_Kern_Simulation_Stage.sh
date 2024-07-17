#!/bin/bash
#SBATCH --job-name=Optimize_Exploration_1_Kern_Simulation_Stage
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4400
#SBATCH --extra-node-info=2:20:1
#SBATCH --time=32:00:00
#SBATCH --mail-type=FAIL

#SBATCH --array=0-9

#SBATCH --output=%x_%A_%a.out

#SBATCH --begin=now+0hour

task_index_mini=0
task_index_maxi=$SLURM_NTASKS_PER_NODE

task_index_start=$((SLURM_ARRAY_TASK_ID*task_index_maxi+task_index_mini))
task_index_final=$(((SLURM_ARRAY_TASK_ID+1)*task_index_maxi+task_index_mini-1))

for task_index in $(seq $task_index_start $task_index_final); do
    task_name="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${task_index}.out"
    task="./Script_Bank/Prime/${SLURM_JOB_NAME}.py" # Careful! The user must provide the path to its own HPC directory!
    python $task $task_index >& $task_name &
done

wait
