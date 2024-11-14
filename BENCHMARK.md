# BENCHMARK for LocalChat

### Lightweighted LLM ability in 0.5b-3b

##### Noise Rate: 0.0

| Model          | Dataset | Accuracy              | Time   |
|----------------|---------|-----------------------|--------|
| qwen2:0.5b     | en      | 0.8167                | 05:37  |
| qwen2:0.5b     | zh      | 0.8033                | 03:45  |
| qwen2:1.5b     | en      | 0.8633                | 02:42  |
| qwen2:1.5b     | zh      | 0.6767                | 02:45  |
| llama3.2:1b    | en      | **0.9000**            | 02:58  |
| llama3.2:1b    | zh      | 0.8033                | 05:12  |
| gemma2:2b      | en      | **0.9500**            | 04:50  |
| gemma2:2b      | zh      | **0.9333**            | 05:56  |

##### Noise Rate: 0.4

| Model          | Dataset | Accuracy              | Time   |
|----------------|---------|-----------------------|--------|
| qwen2:0.5b     | en      | 0.7400                | 05:57  |
| qwen2:0.5b     | zh      | 0.7167                | 03:42  |
| qwen2:1.5b     | en      | 0.7967                | 02:51  |
| qwen2:1.5b     | zh      | 0.6167                | 02:33  |
| llama3.2:1b    | en      | **0.8800**            | 03:41  |
| llama3.2:1b    | zh      | 0.6933                | 05:58  |
| gemma2:2b      | en      | **0.9100**            | 05:14  |
| gemma2:2b      | zh      | **0.9067**            | 06:14  |


##### Noise Rate: 0.8

| Model          | Dataset | Accuracy              | Time   |
|----------------|---------|-----------------------|--------|
| qwen2:0.5b     | en      | 0.4667                | 06:13  |
| qwen2:0.5b     | zh      | 0.4767                | 04:07  |
| qwen2:1.5b     | en      | 0.5500                | 03:19  |
| qwen2:1.5b     | zh      | 0.4533                | 02:52  |
| llama3.2:1b    | en      | **0.6633**            | 04:08  |
| llama3.2:1b    | zh      | 0.4200                | 07:01  |
| gemma2:2b      | en      | **0.7967**            | 06:02  |
| gemma2:2b      | zh      | **0.7300**            | 06:29  |

##### Summary

| Model          | Dataset | Noise Rate | Accuracy          | Time   |
|----------------|---------|------------|-------------------|--------|
| qwen2:0.5b     | en      | 0.0        | 0.8167            | 05:37  |
| qwen2:0.5b     | en      | 0.4        | 0.7400            | 05:57  |
| qwen2:0.5b     | en      | 0.8        | 0.4667            | 06:13  |
| qwen2:0.5b     | zh      | 0.0        | 0.8033            | 03:45  |
| qwen2:0.5b     | zh      | 0.4        | 0.7167            | 03:42  |
| qwen2:0.5b     | zh      | 0.8        | 0.4767            | 04:07  |
| qwen2:1.5b     | en      | 0.0        | 0.8633            | 02:42  |
| qwen2:1.5b     | en      | 0.4        | 0.7967            | 02:51  |
| qwen2:1.5b     | en      | 0.8        | 0.5500            | 03:19  |
| qwen2:1.5b     | zh      | 0.0        | 0.6767            | 02:45  |
| qwen2:1.5b     | zh      | 0.4        | 0.6167            | 02:33  |
| qwen2:1.5b     | zh      | 0.8        | 0.4533            | 02:52  |
| llama3.2:1b    | en      | 0.0        | 0.9000            | 02:58  |
| llama3.2:1b    | en      | 0.4        | 0.8800            | 03:41  |
| llama3.2:1b    | en      | 0.8        | 0.6633            | 04:08  |
| llama3.2:1b    | zh      | 0.0        | 0.8033            | 05:12  |
| llama3.2:1b    | zh      | 0.4        | 0.6933            | 05:58  |
| llama3.2:1b    | zh      | 0.8        | 0.4200            | 07:01  |
| gemma2:2b      | en      | 0.0        | 0.9500            | 04:50  |
| gemma2:2b      | en      | 0.4        | 0.9100            | 05:14  |
| gemma2:2b      | en      | 0.8        | 0.7967            | 06:02  |
| gemma2:2b      | zh      | 0.0        | 0.9333            | 05:56  |
| gemma2:2b      | zh      | 0.4        | 0.9067            | 06:14  |
| gemma2:2b      | zh      | 0.8        | 0.7300            | 06:29  |

> Due to the severe hallucination of small models, their performance on datasets such as mmlu is generally poor. Therefore, mmlu is not used as an indicator to evaluate the ability of small models. Local small models mainly focus on summarization ability, so the simple and easy-to-use RGB is chosen as the evaluation indicator. The time here is based on 3080.
>
> Ref: RGB https://arxiv.org/pdf/2309.01431