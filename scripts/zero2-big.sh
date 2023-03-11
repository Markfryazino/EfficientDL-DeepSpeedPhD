# запуск большой модели с zero-2 и оффлоадом оптимизатора
deepspeed \
    --include localhost:2,3 \
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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512

deepspeed \
    --include localhost:2,3 \
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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --learning_rate 2e-5 \
    --seed 42 \
    --output_dir ./log \
    --max_examples 512