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

python main_app.py \
--dataset_name weissmann_rpe1 \
--output_directory /home/mila/s/siba-smarak.panigrahi/scratch/cbc/causalbench_final_output \
--data_directory  /home/mila/s/siba-smarak.panigrahi/scratch/cbc/data \
--training_regime partial_interventional \
--partial_intervention_seed 0 \
--fraction_partial_intervention 0.5 \
--subset_data 0.1 \
--model_seed 0 \
--model_name custom \
--inference_function_file_path dcdfg.py \
--do_filter \
--gene_partition_sizes 282 \
--soft_adjacency_matrix_threshold 0.2 \
--reg_coeff 0.0 \
--num_modules 10



python scripts/plots.py /home/mila/l/lena-nehale.ezzine/scratch/causalbench/plots /home/mila/l/lena-nehale.ezzine/scratch/causalbench/cbc_output