causalbench_run \
--dataset_name weissmann_rpe1 \
--output_directory /home/mila/l/lena-nehale.ezzine/scratch/causalbench/cbc_output \
--data_directory /home/mila/l/lena-nehale.ezzine/scratch/causalbench/cbc_data \
--training_regime partial_interventional \
--partial_intervention_seed 0 \
--fraction_partial_intervention 0.5 \
--model_name DCDI-DSF \
--subset_data 1 \
--model_seed 0 \
--do_filter

causalbench_run \
--dataset_name weissmann_rpe1 \
--output_directory /home/mila/l/lena-nehale.ezzine/scratch/causalbench/cbc_output \
--data_directory  /home/mila/l/lena-nehale.ezzine/scratch/causalbench/cbc_data \
--training_regime partial_interventional \
--partial_intervention_seed 0 \
--fraction_partial_intervention 0.5 \
--model_name custom \
--inference_function_file_path dcdfg.py \
--subset_data 0.05 \
--model_seed 0 \
--do_filter

python scripts/plots.py /home/mila/l/lena-nehale.ezzine/scratch/causalbench/plots /home/mila/l/lena-nehale.ezzine/scratch/causalbench/cbc_output