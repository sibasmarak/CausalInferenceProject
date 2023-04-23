causalbench_run \
--dataset_name weissmann_rpe1 \
--output_directory /home/mila/s/siba-smarak.panigrahi/scratch/cbc/causalbench_output \
--data_directory /home/mila/s/siba-smarak.panigrahi/scratch/cbc/data \
--training_regime partial_interventional \
--partial_intervention_seed 0 \
--fraction_partial_intervention 0.5 \
--model_name dcdi \
--subset_data 1 \
--model_seed 0 \
--do_filter

causalbench_run \
--dataset_name weissmann_rpe1 \
--output_directory /home/mila/s/siba-smarak.panigrahi/scratch/cbc/causalbench_output \
--data_directory  /home/mila/s/siba-smarak.panigrahi/scratch/cbc/data \
--training_regime partial_interventional \
--partial_intervention_seed 0 \
--fraction_partial_intervention 0.5 \
--model_name DCDI-G \
--inference_function_file_path dcdi.py \
--subset_data 1 \
--model_seed 0 \
--do_filter

python scripts/plots.py /home/mila/s/siba-smarak.panigrahi/scratch/cbc/plots /home/mila/s/siba-smarak.panigrahi/scratch/cbc/causalbench_output