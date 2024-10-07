#!/bin/bash
SESSION_NAME="Fragile_Baseline"
NUM_WINDOWS=4
gpu_path="/nfs/stak/users/brownhun/hpc-share/Tactile_Benchmark/learning/HPC/hpc_launch.bash"
tmux new-session -d -s "$SESSION_NAME"

for i in $(seq 1 $NUM_WINDOWS); do
    tmux new-window -t "$SESSION_NAME":$i -n "Window $i"
    tmux send-keys -t "$SESSION_NAME":$i "conda activate mani && "
    tmux send-keys -t "$SESSION_NAME":$i "bash $gpu_path $i $i " 
    tmux send-keys -t "$SESSION_NAME":$i "$*"
    tmux send-keys -t "$SESSION_NAME":$i "; exit" C-m
    #tmux send-keys -t "$SESSION_NAME":$i "echo $i $i; sleep 2; exit" C-m
done

window_count=$(tmux list-windows | wc -l)
while [ $window_count -gt 0 ]; do
    sleep 60
    window_count=$(tmux list-windows | wc -l)
    if [ "$window_count" -eq 1 ]; then
        # Close the tmux session
        window_count=0
        tmux kill-session
    fi
done
#tmux attach -t "$SESSION_NAME"