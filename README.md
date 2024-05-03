## MADLLM: Multivariate Anomaly Detection for Cloud Systems via Pre-trained LLMs


## Get Start
- Create a new environment using conda. `conda create -n madllm python=3.8`
- Activate the environment. `conda activate madllm`
- Install the required packages. `pip install -r requirements`
- Download data. You can obtain all the benchmarks from [[here](https://drive.google.com/drive/folders/1dYba8m2W0LWfh6btAjdOphi_dWlbd9lB?usp=sharing)]. You can make a dir called `all_datasets`, put the dir into the root path of MADLLM, and download the benchmarks into the dir.


## Train and Test
We provide the experiment scripts of all benchmarks under the folder `./scripts`. For example, to run the test on dataset MSL, just run `bash ./scripts/MSL.sh`. You can change the configuration of the experiment in the script. Here are explanation of some parameters:
```bash
--use_skip_embedding     whether use skip embedding
--top_k                  the prompt selected in the sub-pool
--prompt_len             the length of each prompt value
--nb_random_samples      the number of negative patches 
```

We use the pre-trained GPT2 model as the LLM model. If you cannot download the LLM model from huggingface, you can also download it from [[here](https://drive.google.com/drive/folders/1TgGm9DpnF1HHRV7eMPUuRupt0hyr4eVq?usp=sharing)].



## Test from Pre-trained Checkpoints
We also provide the pre-trained checkpoints of MADLLM in [[here]]. For example, to run the experiment of MSL dataset from checkpoint, just write the configuration `--training = 0` into `./scripts/MSL.sh` and run `bash ./scripts/MSL.sh`