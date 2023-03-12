# запуск большой модели с zero-2 и оффлоадом оптимизатора и модели
# 3 разных батч сайза: 32, 64 и 128
deepspeed \
    --include localhost:0,1 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero2-offload.json \
    --model_name bigscience/bloomz-3b \
    --wandb_run_name zero2-big-32 \
    --fp16 \
    --num_train_epochs 1 \
    --do_train \
    --evaluation_strategy no \
    --save_strategy no \
    --logging_steps 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512

deepspeed \
    --include localhost:0,1 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero2-offload.json \
    --model_name bigscience/bloomz-3b \
    --wandb_run_name zero2-big-64 \
    --fp16 \
    --num_train_epochs 1 \
    --do_train \
    --evaluation_strategy no \
    --save_strategy no \
    --logging_steps 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512

deepspeed \
    --include localhost:0,1 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero2-offload.json \
    --model_name bigscience/bloomz-3b \
    --wandb_run_name zero2-big-128 \
    --fp16 \
    --num_train_epochs 1 \
    --do_train \
    --evaluation_strategy no \
    --save_strategy no \
    --logging_steps 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512