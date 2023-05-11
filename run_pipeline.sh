#!/bin/bash
#SBATCH --job-name=cbc-dcdfg
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=50
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7

module load anaconda/3
conda activate cbc

cd /home/mila/s/siba-smarak.panigrahi/causalbench-starter-iclr/src

# Set up paths and parameters
DATASET_NAME="weissmann_rpe1"
OUTPUT_DIRECTORY="/home/mila/s/siba-smarak.panigrahi/scratch/cbc/causalbench_output"
DATA_DIRECTORY="/home/mila/s/siba-smarak.panigrahi/scratch/cbc/data"
TRAINING_REGIME="partial_interventional"
PARTIAL_INTERVENTION_SEED=0
MODEL_NAME="custom"
INFERENCE_FUNCTION_FILE_PATH="dcdfg.py"
SUBSET_DATA=1.0
MODEL_SEED=0
PLOT_DIRECTORY="/home/mila/s/siba-smarak.panigrahi/scratch/cbc/plots"

adj_thresholds=(0.2 0.5)
modules=(20 10)
sparsities=(0.0 0.1)

GENE_PARTITION_SIZE=282
ADJ_THRESHOLD=${adj_thresholds[$(($((SLURM_ARRAY_TASK_ID/4))%2))]} 
MODULES=${modules[$(($((SLURM_ARRAY_TASK_ID/2))%2))]} 
SPARSITY=${sparsities[$((SLURM_ARRAY_TASK_ID%2))]} 

echo "ADJ_THRESHOLD: ${ADJ_THRESHOLD}"
echo "MODULES: ${MODULES}"
echo "SPARSITY: ${SPARSITY}"


# Loop over the different fraction values
for FRACTION in 0.25 0.5 0.75 1.0; do
    # Construct the command
    COMMAND="python main_app.py \
        --dataset_name ${DATASET_NAME} \
        --output_directory ${OUTPUT_DIRECTORY} \
        --data_directory ${DATA_DIRECTORY} \
        --training_regime ${TRAINING_REGIME} \
        --partial_intervention_seed ${PARTIAL_INTERVENTION_SEED} \
        --fraction_partial_intervention ${FRACTION} \
        --model_name ${MODEL_NAME} \
        --inference_function_file_path ${INFERENCE_FUNCTION_FILE_PATH} \
        --subset_data ${SUBSET_DATA} \
        --model_seed ${MODEL_SEED} \
        --do_filter \
        --gene_partition_sizes ${GENE_PARTITION_SIZE} \
        --soft_adjacency_matrix_threshold ${ADJ_THRESHOLD} \
        --reg_coeff ${SPARSITY} \
        --num_modules ${MODULES}"

    # Run the command
    echo "Running command: ${COMMAND}"
    eval ${COMMAND}
done

# Generate plots
cd ..
PLOTS_SCRIPT="python scripts/plots.py ${PLOT_DIRECTORY}  ${OUTPUT_DIRECTORY}"
echo "Running plots script: ${PLOTS_SCRIPT}"
eval ${PLOTS_SCRIPT}
