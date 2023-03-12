# запуск мелкой модели без zero
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 29501 \
    run.py \
    --model_name bigscience/bloomz-3b \
    --wandb_run_name fsdp-full-big-64 \
    --fp16 \
    --num_train_epochs 1 \
    --do_train \
    --evaluation_strategy no \
    --save_strategy no \
    --logging_steps 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512
    --fsdp "full_shard offload"

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 29501 \
    run.py \
    --model_name bigscience/bloomz-3b \
    --wandb_run_name fsdp-grad-big-64 \
    --fp16 \
    --num_train_epochs 1 \
    --do_train \
    --evaluation_strategy no \
    --save_strategy no \
    --logging_steps 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512
    --fsdp "shard_grad_op offload"