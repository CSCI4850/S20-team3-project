#!/usr/bin/env bash

printf "Make sure you have python3.7 and virtualenv installed before continuing! Press enter to continue, or Ctrl-C to quit."
read tmp

printf "Absolute path to python3.7: "
read pypath

virtualenv --python=$pypath venv

source venv/bin/activate

pip install -r requirements.txt

# Write updated conditions and RAM maps to respective files

cat <<EOF > venv/lib/python3.7/site-packages/retro/data/stable/GalagaDemonsOfDeath-Nes/scenario.json
{
  "done": {
    "variables": {
      "stage": {
        "op": "equal",
        "reference": 0
      }
    }
  },
  "reward": {
    "variables": {
      "score": {
        "reward": 1
      }
    }
  }
}
EOF

cat <<EOF > venv/lib/python3.7/site-packages/retro/data/stable/GalagaDemonsOfDeath-Nes/data.json
{
  "info": {
    "lives": {
      "address": 1159,
      "type": "|u1"
    },
    "score": {
      "address": 225,
      "type": ">n6"
    },
    "stage": {
      "address": 1155,
      "type": "|u1"
    }
  }
}
EOF

python -m retro.import .

printf "All done! Use\n\nsource venv/bin/activate\n\nto enter the environment, or run demo.sh to start a 5 episode demo of training the Uniform Single Q-Learning network!"
