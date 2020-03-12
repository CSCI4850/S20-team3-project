# S20-team3-project

Required Packages:

    Python     == 3.7
    gym-retro  >= 0.7.0
    tensorflow == 2.1.0
    
And all their dependencies. These are available in the requirements.txt

Create a 3.7 virtualenv

    virtualenv --python=/path/to/python3.7 venv # /usr/bin/python3.7 for example

start the virtual environment- ensure that you are in your local repository

    source venv/bin/activate
    
install requirements:
    
    pip3 install -r requirements.txt
    
import the rom:

    python3 -m retro.import 'Galaga - Demons of Death (USA).nes'

after virtual environment is active, step into either SingleQ/Uniform/ or
SingleQ/Prioritized/ and use:

    ./Galaga.py
    
to watch the gameplay, simply use:

    ./Galaga.py --play

## Action Space
As designated by Arcade Learning Environment Technical Manual[^1], we have
selected five possible actions to control the game:

- Fire       (1)
- Right      (2)
- Right-Fire (3)
- Left       (4)
- Left-Fire  (5)

This exists as our "small action space", where we restrict the network's
available actions to prevent it from "gaming" the reward system or getting
stuck. However, the default action space for Galaga is also the same as this
small action space, simply repeating as per the atari-py implementation manual[^1].

## References
[1]: https://github.com/openai/atari-py/blob/master/doc/manual/manual.pdf
