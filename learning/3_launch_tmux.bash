#!/bin/bash

# Create a new tmux session
tmux new-session -d -s "Fragile_Baseline"

# Window 1: Run script1.sh
tmux send-keys -t "Fragile_Baseline:0" "bash learning/launch.bash 1 3" C-m

# Create a new window and run script2.sh
tmux new-window -t "Fragile_Baseline" -n "Window 2" "bash learning/launch.bash 4 6"

# Create another window and run script3.sh
tmux new-window -t "Fragile_Baseline" -n "Window 3" "bash learning/launch.bash 7 10"

# Attach to the session
tmux attach -t "Fragile_Baseline"
