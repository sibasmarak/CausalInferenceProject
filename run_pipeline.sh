#!/bin/bash
#SBATCH --job-name=cbc-dcdi
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=50
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0

module load anaconda/3
conda activate cbc

# Set up paths and parameters
DATASET_NAME="weissmann_rpe1"
OUTPUT_DIRECTORY="/home/mila/s/siba-smarak.panigrahi/scratch/cbc/causalbench_output"
DATA_DIRECTORY="/home/mila/s/siba-smarak.panigrahi/scratch/cbc/data"
TRAINING_REGIME="partial_interventional"
PARTIAL_INTERVENTION_SEED=0
MODEL_NAME="DCDI-G"
INFERENCE_FUNCTION_FILE_PATH="dcdi.py"
SUBSET_DATA=1.0
MODEL_SEED=0
PLOT_DIRECTORY="/home/mila/s/siba-smarak.panigrahi/scratch/cbc/plots"

# Loop over the different fraction values
for FRACTION in 0.25 0.5 0.75 1.0; do
    # Construct the command
    COMMAND="causalbench_run \
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
        --do_filter"

    # Run the command
    echo "Running command: ${COMMAND}"
    eval ${COMMAND}
done

# Generate plots
PLOTS_SCRIPT="python scripts/plots.py ${PLOT_DIRECTORY}  ${OUTPUT_DIRECTORY}"
echo "Running plots script: ${PLOTS_SCRIPT}"
eval ${PLOTS_SCRIPT}
