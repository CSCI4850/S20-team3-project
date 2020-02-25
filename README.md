# S20-team3-project

Required Packages:

    Python    == 3.7
    gym-retro >= 0.7.0
    
And all their dependencies

Create a 3.7 virtualenv

    virtualenv --python=/path/to/python3.7 venv # /usr/bin/python3.7 for example

start the virtual environment- ensure that you are in your local repository

    source venv/bin/activate
    
import the rom:

    python3 -m retro.import 'Galaga - Demons of Death (USA).nes'

after virtual environment is active:

    ./basic.py
    

## Action Space
As designated by Arcade Learning Environment Technical Manual[^1], we have
selected five possible actions to control the game:

- Fire       (1)
- Right      (2)
- Left       (3)
- Right-Fire (11)
- Left-Fire  (12)

## References
[^1]: https://github.com/openai/atari-py/blob/master/doc/manual/manual.pdf
