import os
import wandb

from src.pipeline.pipe_train import PipelineTrainer

# Login to WB with API key
os.environ['WANDB_API_KEY'] = "666eab241c17c161a8d9e0ca319adf27b102e99c"
wandb.login()

# Load pipeline
p = PipelineTrainer(
    config_path='configs/pipeline/train__musicgen__gtzan.json'
)

# Run pipeline
p.run_pipeline(push_model_to_hf=False)