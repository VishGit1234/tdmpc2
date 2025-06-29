#!/bin/bash
#SBATCH --job-name=rl_research
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres tmpdisk:40480,shard:10000
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j-%x.out  # %j: job ID, %x: job name. Reference: https://slurm.schedmd.com/sbatch.html#lbAH
 
slurm-start-dockerd.sh
export DOCKER_HOST=unix:///tmp/run/docker.sock
docker pull ghcr.io/vishgit1234/tdmpc2:latest
docker compose build
docker compose up
docker push ghcr.io/vishgit1234/tdmpc2:latest