#!/bin/bash

# start end env_id obs_mode dmg_force
# Create a new tmux session
tmux new-session -d -s "Fragile_Baseline"
pid=$!
# Window 1: Run script1.sh
tmux send-keys -t "Fragile_Baseline:0" "conda activate mani && "
tmux send-keys -t "Fragile_Baseline:0" "bash learning/launch.bash 1 3 $1 $2 $3" C-m

# Create a new window and run script2.sh
tmux new-window -t "Fragile_Baseline" -n "Window 2" "bash learning/launch.bash 4 6 $1 $2 $3"

# Create another window and run script3.sh
tmux new-window -t "Fragile_Baseline" -n "Window 3" "bash learning/launch.bash 7 10 $1 $2 $3; tmux kill-session"

# Attach to the session
tmux attach -t "Fragile_Baseline"

wait $pid
echo done