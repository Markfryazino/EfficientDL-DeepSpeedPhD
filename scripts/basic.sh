deepspeed \
    --include localhost:2,3 \
    run.py \
    --deepspeed deepspeed/basic.json \
    --model_name bigscience/bloom-560m \
    --run_name test \
    --fp16 \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --per_device_train_batch_size 1 \
    --learning-rate 2e-5 \
    --seed 42 \
    --report-to wandb