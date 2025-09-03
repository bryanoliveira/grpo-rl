## Without Docker

1. Install UV.
2. Install dependencies with `uv sync` and then `uv sync --extra mujoco`
3. Test with CartPole: `uv run python algorithm.py --no-track`
4. Run all missing experiments using all CPU cores: `bash launch_all_cpus.sh`
6. Alternatively, manually run the experiments with `bash experiment.sh <total_instances> <current_instance_index>`, e.g.:

```bash
bash experiment.sh 1 0  # to run all experiments sequentially
```

or 

```bash
# each in a different terminal instance (e.g. tmux):
bash experiment.sh 4 0
bash experiment.sh 4 1  
bash experiment.sh 4 2
bash experiment.sh 4 3
```

The `experiment.sh` script will first enumerate all experiments and then split them into the total number of instances and run the commands that are multiples of the current instance index.

## With Docker

1. `docker build -t grpo .`
2. `docker run -e WANDB_API_KEY=<key> grpo`
