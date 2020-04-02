#!/usr/bin/env bash

cat <<EOF
Welcome to the short demo for our Galaga neural network utilizing the Deep Q-Learning architecture!
In a moment, the python file located at the SingleQ/Uniform/Galaga.py location will execute and run for 5 episodes.
Be warned, this network can be resource intensive in both RAM and CPU or GPU, depending on your setup.
We hope you enjoy!
EOF

sleep 5

cd SingleQ/Uniform

./Galaga.py --play
