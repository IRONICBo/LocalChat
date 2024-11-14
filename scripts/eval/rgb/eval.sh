#!/bin/bash

python eval.py --ollamamodel qwen2:0.5b --dataset en --noise_rate 0.0
# 0.8166666666666667 / 05:37
python eval.py --ollamamodel qwen2:0.5b --dataset en --noise_rate 0.4
# 0.74 / 05:57
python eval.py --ollamamodel qwen2:0.5b --dataset en --noise_rate 0.8
# 0.4666666666666667 / 06:13
python eval.py --ollamamodel qwen2:0.5b --dataset zh --noise_rate 0.0
# 0.8033333333333333 / 03:45
python eval.py --ollamamodel qwen2:0.5b --dataset zh --noise_rate 0.4
# 0.7166666666666667 / 03:42
python eval.py --ollamamodel qwen2:0.5b --dataset zh --noise_rate 0.8
# 0.4766666666666667 / 04:07

python eval.py --ollamamodel qwen2:1.5b --dataset en --noise_rate 0.0
# 0.8633333333333333  / 02:42
python eval.py --ollamamodel qwen2:1.5b --dataset en --noise_rate 0.4
# 0.7966666666666666  / 02:51
python eval.py --ollamamodel qwen2:1.5b --dataset en --noise_rate 0.8
# 0.55 / 03:19
python eval.py --ollamamodel qwen2:1.5b --dataset zh --noise_rate 0.0
# 0.676666666666666 / 02:45
python eval.py --ollamamodel qwen2:1.5b --dataset zh --noise_rate 0.4
# 0.6166666666666667 / 02:33
python eval.py --ollamamodel qwen2:1.5b --dataset zh --noise_rate 0.8
# 0.4533333333333333 / 02:52

python eval.py --ollamamodel llama3.2:1b --dataset en --noise_rate 0.0
# 0.9 / 02:58
python eval.py --ollamamodel llama3.2:1b --dataset en --noise_rate 0.4
# 0.88 / 03:41
python eval.py --ollamamodel llama3.2:1b --dataset en --noise_rate 0.8
# 0.6633333333333333 / 04:08
python eval.py --ollamamodel llama3.2:1b --dataset zh --noise_rate 0.0
# 0.8033333333333333 / 05:12
python eval.py --ollamamodel llama3.2:1b --dataset zh --noise_rate 0.4
# 0.6933333333333334 / 05:58
python eval.py --ollamamodel llama3.2:1b --dataset zh --noise_rate 0.8
# 0.42 / 07:01

python eval.py --ollamamodel gemma2:2b --dataset en --noise_rate 0.0
# 0.95 / 04:50
python eval.py --ollamamodel gemma2:2b --dataset en --noise_rate 0.4
# 0.91 / 05:14
python eval.py --ollamamodel gemma2:2b --dataset en --noise_rate 0.8
# 0.7966666666666666 / 06:02
python eval.py --ollamamodel gemma2:2b --dataset zh --noise_rate 0.0
# 0.9333333333333333 / 05:56
python eval.py --ollamamodel gemma2:2b --dataset zh --noise_rate 0.4
# 0.9066666666666666 / 06:14
python eval.py --ollamamodel gemma2:2b --dataset zh --noise_rate 0.8
# 0.73 / 06:29