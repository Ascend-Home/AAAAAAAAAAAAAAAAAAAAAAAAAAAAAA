# MAX-AI: Frontier-Scale LLM From Scratch

1.5T parameter MoE transformer. MLA + MoE + MTP + 1M context.

## Requirements
- 25,000+ H100 GPUs
- 30T tokens of training data
- ~$400M budget
- Will probably fail. That's the point.

## Quickstart
```bash
docker compose up --build
bash scripts/launch_cluster.sh
