# запуск большой модели с zero-3 и оффлоадом оптимизатора и модели
# 3 разных батч сайза: 32, 64 и 128
deepspeed \
    --include localhost:2,3 \
    --master_port 29502 \
    run.py \
    --deepspeed deepspeed/zero3-offload.json \
    --model_name bigscience/bloomz-3b \
    --wandb_run_name zero3-big-32 \
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
    --include localhost:2,3 \
    --master_port 29502 \
    run.py \
    --deepspeed deepspeed/zero3-offload.json \
    --model_name bigscience/bloomz-3b \
    --wandb_run_name zero3-big-64 \
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
    --include localhost:2,3 \
    --master_port 29502 \
    run.py \
    --deepspeed deepspeed/zero3-offload.json \
    --model_name bigscience/bloomz-3b \
    --wandb_run_name zero3-big-128 \
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