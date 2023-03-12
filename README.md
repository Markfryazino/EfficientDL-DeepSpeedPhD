# Efficient DL Systems, Homework 6
## PhD по DeepSpeed

Ну вообще я бы не назвал это прям PhD, скорее что-то вроде специалитета.

Проект W&B [тут](https://wandb.ai/broccoliman/efficient_dl_deepspeed_phd). Отчёт [тут](https://wandb.ai/broccoliman/efficient_dl_deepspeed_phd/reports/Testing-DeepSpeed-HF--VmlldzozNzYyNTkx).

### Как запускать
```
git clone https://github.com/Markfryazino/EfficientDL-DeepSpeedPhD.git
pip install -r requirements.txt
export WANDB_API_KEY=<your secret key>
bash scripts/basic-small.sh
```

### Структура
Всё происходит в скрипте [run.py](run.py). Там скачивается модель и датасет (SST-2) и через Trainer запускается обучение. Вообще, вместо него можно было бы использовать [run_glue.py из документации](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py), но своё написать всегда приятнее.

В папке [scripts](scripts/) лежат bash-скрипты, которые можно запускать. А в папке [deepspeed](deepspeed/) лежат разные конфиги дипспида.
