#!/bin/bash

# Run all evaluation tasks in parallel
# Each task runs in the background (&), using different GPU devices

echo "Starting parallel evaluation..."

# Task 1: 1_en-zh_greedy_all
python before_lexical.py --input_folder ../1_en-zh_greedy_all --output_folder ../evaluation_results_lexical/1_en-zh_greedy_all_lexical --device cuda:0 --skip_existing True &
PID1=$!
echo "Task 1 (1_en-zh_greedy_all) started with PID: $PID1 on cuda:0"

# Task 2: 10_en-de_sampling_0.5_5models
python before_lexical.py --input_folder ../10_en-de_sampling_0.5_5models --output_folder ../eval_results_lexical/10_en-de_sampling_0.5_5models_lexical --device cuda:0 --skip_existing True &
PID2=$!
echo "Task 2 (10_en-de_sampling_0.5_5models) started with PID: $PID2 on cuda:0"

# Task 3: 10_en-ru_sampling_0.5_5models
python before_lexical.py --input_folder ../10_en-ru_sampling_0.5_5models --output_folder ../evaluation_results_lexical/10_en-ru_sampling_0.5_5models_lexical --device cuda:1 --skip_existing True &
PID3=$!
echo "Task 3 (10_en-ru_sampling_0.5_5models) started with PID: $PID3 on cuda:1"

# Task 4: 10_en-zh_sampling_0.5_all
python before_lexical.py --input_folder ../10_en-zh_sampling_0.5_all --output_folder ../evaluation_results_lexical/10_en-zh_sampling_0.5_all_lexical --device cuda:1 --skip_existing True &
PID4=$!
echo "Task 4 (10_en-zh_sampling_0.5_all) started with PID: $PID4 on cuda:1"

# Task 5: 20_en-zh_sampling_0.5_5models
python before_lexical.py --input_folder ../20_en-zh_sampling_0.5_5models --output_folder ../evaluation_results_lexical/20_en-zh_sampling_0.5_5models_lexical --device cuda:2 --skip_existing True &
PID5=$!
echo "Task 5 (20_en-zh_sampling_0.5_5models) started with PID: $PID5 on cuda:2"

# Task 6: 50_en-zh_sampling_0.5_5models
python before_lexical.py --input_folder ../50_en-zh_sampling_0.5_5models --output_folder ../evaluation_results_lexical/50_en-zh_sampling_0.5_5models_lexical --device cuda:2 --skip_existing True &
PID6=$!
echo "Task 6 (50_en-zh_sampling_0.5_5models) started with PID: $PID6 on cuda:2"

# Wait for all background tasks to complete
echo ""
echo "Waiting for all tasks to complete..."
wait $PID1 && echo "✓ Task 1 (1_en-zh_greedy_all) completed" || echo "✗ Task 1 (1_en-zh_greedy_all) failed"
wait $PID2 && echo "✓ Task 2 (10_en-de_sampling_0.5_5models) completed" || echo "✗ Task 2 (10_en-de_sampling_0.5_5models) failed"
wait $PID3 && echo "✓ Task 3 (10_en-ru_sampling_0.5_5models) completed" || echo "✗ Task 3 (10_en-ru_sampling_0.5_5models) failed"
wait $PID4 && echo "✓ Task 4 (10_en-zh_sampling_0.5_all) completed" || echo "✗ Task 4 (10_en-zh_sampling_0.5_all) failed"
wait $PID5 && echo "✓ Task 5 (20_en-zh_sampling_0.5_5models) completed" || echo "✗ Task 5 (20_en-zh_sampling_0.5_5models) failed"
wait $PID6 && echo "✓ Task 6 (50_en-zh_sampling_0.5_5models) completed" || echo "✗ Task 6 (50_en-zh_sampling_0.5_5models) failed"

echo ""
echo "All evaluation tasks completed!"

