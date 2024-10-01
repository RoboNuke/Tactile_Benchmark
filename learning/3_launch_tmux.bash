#!/bin/bash

# start end env_id obs_mode dmg_force exp_name
# Create a new tmux session
tmux new-session -d -s "Fragile_Baseline"

# Window 1: Run script1.sh
tmux send-keys -t "Fragile_Baseline:0" "conda activate mani && "
tmux send-keys -t "Fragile_Baseline:0" "bash learning/launch.bash 1 2 $1 $2 $3 $4; exit" C-m

# Create a new window and run script2.sh
tmux split-window -v
tmux send-keys -t 1 "conda activate mani && bash learning/launch.bash 3 4 $1 $2 $3 $4; sleep 60; exit" C-m

# Create another window and run script3.sh
tmux split-window -v
tmux send-keys -t 2 "conda activate mani && bash learning/launch.bash 5 6 $1 $2 $3 $4; sleep 60; exit" C-m

# Attach to the session
tmux attach -t "Fragile_Baseline"
#while tmux list-sessions | grep -q "Fragile_Baseline"; do
#    sleep 600 # check every 10 minutes
#done
#echo "   " $1 $2 $3 $4 complete!