# запуск мелкой модели на 1 gpu с оффлоадингом
deepspeed \
    --include localhost:0 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero2.json \
    --model_name bigscience/bloom-560m \
    --wandb_run_name 1gpu-small-no-offload \
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
    --include localhost:0 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero2-offload-opt.json \
    --model_name bigscience/bloom-560m \
    --wandb_run_name 1gpu-small-offload-opt \
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
    --include localhost:0 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero2-offload-model.json \
    --model_name bigscience/bloom-560m \
    --wandb_run_name 1gpu-small-offload-model \
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
    --include localhost:0 \
    --master_port 29501 \
    run.py \
    --deepspeed deepspeed/zero2-offload.json \
    --model_name bigscience/bloom-560m \
    --wandb_run_name 1gpu-small-offload-all \
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