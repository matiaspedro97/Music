import os
import wandb

from src.pipeline.pipe_train import PipelineTrainer

# Login to WB with API key
wandb.login()

# Load pipeline
p = PipelineTrainer(
    config_path='configs/pipeline/train__musicgen__gtzan.json'
)

# Run pipeline
p.run_pipeline(push_model_to_hf=True)