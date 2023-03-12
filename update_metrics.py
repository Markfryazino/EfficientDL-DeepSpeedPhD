import wandb
import numpy as np

from tqdm import tqdm


api = wandb.Api()
for run in tqdm(api.runs('broccoliman/efficient_dl_deepspeed_phd')):
    system_metrics = run.history(stream="events")
    memories = []
    for i in range(4):
        if f"system.gpu.process.{i}.memoryAllocated" in system_metrics.columns:
            memories += system_metrics[f"system.gpu.process.{i}.memoryAllocated"].tolist()
        max_gpu_memory = np.max(memories) if len(memories) > 0 else None
        run.summary.update({"max_GPU_memory": max_gpu_memory})
