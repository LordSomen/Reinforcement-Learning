# Reinforcement Learning Projects

This repository collects reinforcement learning coursework, experiments, and implementation notes. The material is split across assignment folders, with notebooks, training scripts, checkpoints, plots, and small demo/playback utilities.

## Repository Layout

- `Assignment-1A/` - introductory notebook-based work
- `Assignment-1B/` - Tic-Tac-Toe implementation and notebook
- `Assignment-3-final/` - larger RL project covering DQN and policy-gradient work

The most detailed documentation lives inside the assignment folders, especially `Assignment-3-final/readme.md`.

## What's Inside

Typical content in this repo includes:

- Jupyter notebooks for experimentation and writeups
- Python training scripts for RL agents
- pretrained checkpoints saved as `.pth` files
- plots and reports for training analysis
- simple play/demo scripts for loading trained policies

## Recommended Setup

Most of the code in this repository is Python-based. A typical environment for the larger projects includes:

- Python 3.10+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- OpenCV for image-based environments

If you are working inside one of the assignment folders, check that folder's README or notebook for the exact package list and run commands.

## Running the Projects

There is no single entrypoint for the whole repository. Run the scripts from the relevant assignment folder, for example:

- open the notebook in `Assignment-1A/` or `Assignment-1B/`
- run the training scripts in `Assignment-3-final/mountain_car/`, `Assignment-3-final/pong/`, or `Assignment-3-final/PROBLEM-3/`
- use the play scripts in those folders to load trained checkpoints and watch the learned agents

## Notes

- Many scripts expect paths to checkpoints or assets to be updated before execution.
- Some experiments are expensive to train and may take a long time to finish.
- If you only want the assignment-specific instructions, start with `Assignment-3-final/readme.md`.
