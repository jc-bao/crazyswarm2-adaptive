#!/usr/bin/zsh

export DISPLAY=:1

# 1. Close all the tmux session called ros
echo "Closing all tmux sessions called ros"
tmux kill-session -t ros 

# reboot crazyflie
echo "Rebooting Crazyflie"
zsh ./reboot.sh
# connect attempt
echo "Connecting to Crazyflie"
# mamba activate ros2
python connect.py


# 2. Create a tmux session called ros and split the first window horizontally
echo "Creating a tmux session called ros and split the first window horizontally"
tmux new-session -d -s ros
tmux split-window -h -t ros
# set mouse on
tmux set -g mouse on

# 3. Send the command specified to each panel in the session
echo "Sending the command specified to each panel in the session"
setup_command="cd ~/Research/code/crazyswarm2-adaptive; mamba activate ros2; . ./install/local_setup.zsh"
tmux send-keys -t ros:1.1 "$setup_command" C-m
tmux send-keys -t ros:1.2 "$setup_command" C-m

# send specific commands to each panel
echo "Sending specific commands to each panel"
tmux send-keys -t ros:1.1 "ros2 launch crazyflie launch.py" C-m
tmux send-keys -t ros:1.2 "ros2 run crazyflie_examples cfctl"

# 4. Attach to the session
tmux attach-session -t ros