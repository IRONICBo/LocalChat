#!/bin/bash

python eval.py --ollamamodel qwen2:0.5b --dataset en --noise_rate 0.0
python eval.py --ollamamodel qwen2:0.5b --dataset en --noise_rate 0.4
python eval.py --ollamamodel qwen2:0.5b --dataset en --noise_rate 0.8
python eval.py --ollamamodel qwen2:0.5b --dataset zh --noise_rate 0.0
python eval.py --ollamamodel qwen2:0.5b --dataset zh --noise_rate 0.4
python eval.py --ollamamodel qwen2:0.5b --dataset zh --noise_rate 0.8

python eval.py --ollamamodel qwen2:1.1b --dataset en --noise_rate 0.0
python eval.py --ollamamodel qwen2:1.1b --dataset en --noise_rate 0.4
python eval.py --ollamamodel qwen2:1.1b --dataset en --noise_rate 0.8
python eval.py --ollamamodel qwen2:1.1b --dataset zh --noise_rate 0.0
python eval.py --ollamamodel qwen2:1.1b --dataset zh --noise_rate 0.4
python eval.py --ollamamodel qwen2:1.1b --dataset zh --noise_rate 0.8

python eval.py --ollamamodel llama3.2:1b --dataset en --noise_rate 0.0
python eval.py --ollamamodel llama3.2:1b --dataset en --noise_rate 0.4
python eval.py --ollamamodel llama3.2:1b --dataset en --noise_rate 0.8
python eval.py --ollamamodel llama3.2:1b --dataset zh --noise_rate 0.0
python eval.py --ollamamodel llama3.2:1b --dataset zh --noise_rate 0.4
python eval.py --ollamamodel llama3.2:1b --dataset zh --noise_rate 0.8

python eval.py --ollamamodel gemma2:2b --dataset en --noise_rate 0.0
python eval.py --ollamamodel gemma2:2b --dataset en --noise_rate 0.4
python eval.py --ollamamodel gemma2:2b --dataset en --noise_rate 0.8
python eval.py --ollamamodel gemma2:2b --dataset zh --noise_rate 0.0
python eval.py --ollamamodel gemma2:2b --dataset zh --noise_rate 0.4
python eval.py --ollamamodel gemma2:2b --dataset zh --noise_rate 0.8