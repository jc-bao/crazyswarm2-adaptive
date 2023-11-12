#!/bin/bash

# 1. Close all the tmux session called ros
tmux kill-session -t ros 

# 2. Create a tmux session called ros and split the first window horizontally
tmux new-session -d -s ros
tmux split-window -h -t ros
# set mouse on
tmux set -g mouse on

# 3. Send the command specified to each panel in the session
setup_command="cd ~/Research/code/crazyswarm2-adaptive; mamba activate humble; . ./install/local_setup.zsh"
tmux send-keys -t ros:1.1 "$setup_command" C-m
tmux send-keys -t ros:1.2 "$setup_command" C-m

# 4. Attach to the session
tmux attach-session -t ros