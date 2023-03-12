# запуск большой модели на 2 gpu с оффлоадингом

deepspeed \
    --include localhost:2,3 \
    --master_port 29503 \
    run.py \
    --deepspeed deepspeed/zero2-offload-opt.json \
    --model_name bigscience/bloomz-3b \
    --wandb_run_name 2gpu-big-offload-opt \
    --fp16 \
    --num_train_epochs 1 \
    --do_train \
    --evaluation_strategy no \
    --save_strategy no \
    --logging_steps 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512

deepspeed \
    --include localhost:2,3 \
    --master_port 29503 \
    run.py \
    --deepspeed deepspeed/zero2-offload.json \
    --model_name bigscience/bloomz-3b \
    --wandb_run_name 2gpu-big-offload-all \
    --fp16 \
    --num_train_epochs 1 \
    --do_train \
    --evaluation_strategy no \
    --save_strategy no \
    --logging_steps 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512