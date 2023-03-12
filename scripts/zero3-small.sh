# запуск мелкой модели с zero-3
deepspeed \
    --include localhost:2,3 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero3.json \
    --model_name bigscience/bloom-560m \
    --wandb_run_name zero3-small \
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
    --max_examples 512

deepspeed \
    --include localhost:2,3 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero3-offload-model.json \
    --model_name bigscience/bloom-560m \
    --wandb_run_name zero3-small-offload-model \
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
    --max_examples 512

deepspeed \
    --include localhost:2,3 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero3-offload-opt.json \
    --model_name bigscience/bloom-560m \
    --wandb_run_name zero3-small-offload-opt \
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
    --max_examples 512

deepspeed \
    --include localhost:2,3 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero3-offload.json \
    --model_name bigscience/bloom-560m \
    --wandb_run_name zero3-small-offload-all \
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
    --max_examples 512