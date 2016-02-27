Arcade Reinforcement Learning
=============================

Python implementation of the reinforcement learning model presented in [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602). The agent is trained in the [Arcade Learning Environment](http://www.arcadelearningenvironment.org).

## Requirements
The [Arcade Learning Environment](http://www.arcadelearningenvironment.org/downloads/) is required for simulating the Atari games. Its python interface should be available in Python, e.g. add it to you `$PYTHONPATH`.
Required Python packages are available in `requirements.txt`. To install, use `pip install -r requirements.txt`.

## Usage
The agent can be run with or without a GUI. Using the GUI will be slow so it is generally preferred for evaluation and demonstration purposes only. The agent stores its state in the `model_data/` folder after a keyboard interrupt (ctrl-c). This data is reloaded by providing the `--continue`(`-c`) flag on run.
To train, run `python simulator.py [--gui] [--continue] path_to_rom`
