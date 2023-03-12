# запуск мелкой модели на fsdp
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 29501 \
    run.py \
    --model_name bigscience/bloom-560m \
    --wandb_run_name fsdp-full-small \
    --fp16 \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --save_strategy no \
    --eval_steps 10 \
    --logging_steps 10 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512 \
    --fsdp full_shard

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 29501 \
    run.py \
    --model_name bigscience/bloom-560m \
    --wandb_run_name fsdp-grad-small \
    --fp16 \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --save_strategy no \
    --eval_steps 10 \
    --logging_steps 10 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512 \
    --fsdp shard_grad_op 