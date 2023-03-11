# запуск мелкой модели с zero-2
deepspeed \
    --include localhost:2,3 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero2.json \
    --model_name bigscience/bloom-560m \
    --wandb_run_name zero2-small \
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
    --output_dir ./log