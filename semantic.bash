#!/bin/bash

# Run all semantic evaluation tasks in parallel
# Each task runs in the background (&), using different GPU devices

echo "Starting parallel semantic evaluation..."

# Task 1: 1_en-zh_greedy_all
CUDA_VISIBLE_DEVICES=0 python before_semantic.py --input ../1_en-zh_greedy_all --output ../evaluation_results_semantic/1_en-zh_greedy_all_semantic --step 10 &
PID1=$!
echo "Task 1 (1_en-zh_greedy_all) started with PID: $PID1 on CUDA:0"

# Task 2: 10_en-de_sampling_0.5_5models
CUDA_VISIBLE_DEVICES=1 python before_semantic.py --input ../10_en-de_sampling_0.5_5models --output ../evaluation_results_semantic/10_en-de_sampling_0.5_5models_semantic --step 10 &
PID2=$!
echo "Task 2 (10_en-de_sampling_0.5_5models) started with PID: $PID2 on CUDA:1"

# Task 3: 10_en-ru_sampling_0.5_5models
CUDA_VISIBLE_DEVICES=2 python before_semantic.py --input ../10_en-ru_sampling_0.5_5models --output ../evaluation_results_semantic/10_en-ru_sampling_0.5_5models_semantic --step 10 &
PID3=$!
echo "Task 3 (10_en-ru_sampling_0.5_5models) started with PID: $PID3 on CUDA:2"

# Task 4: 10_en-zh_sampling_0.5_all
CUDA_VISIBLE_DEVICES=3 python before_semantic.py --input ../10_en-zh_sampling_0.5_all --output ../evaluation_results_semantic/10_en-zh_sampling_0.5_all_semantic --step 10 &
PID4=$!
echo "Task 4 (10_en-zh_sampling_0.5_all) started with PID: $PID4 on CUDA:3"

# Task 5: 20_en-zh_sampling_0.5_5models
CUDA_VISIBLE_DEVICES=0 python before_semantic.py --input ../20_en-zh_sampling_0.5_5models --output ../evaluation_results_semantic/20_en-zh_sampling_0.5_5models_semantic --step 10 &
PID5=$!
echo "Task 5 (20_en-zh_sampling_0.5_5models) started with PID: $PID5 on CUDA:0"

# Task 6: 50_en-zh_sampling_0.5_5models
CUDA_VISIBLE_DEVICES=1 python before_semantic.py --input ../50_en-zh_sampling_0.5_5models --output ../evaluation_results_semantic/50_en-zh_sampling_0.5_5models_semantic --step 10 &
PID6=$!
echo "Task 6 (50_en-zh_sampling_0.5_5models) started with PID: $PID6 on CUDA:1"

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
echo "All semantic evaluation tasks completed!"