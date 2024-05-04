## MADLLM: Multivariate Anomaly Detection for Cloud Systems via Pre-trained LLMs


## Get Start
- Create a new environment using conda. `conda create -n madllm python=3.8`
- Activate the environment. `conda activate madllm`
- Install the required packages. `pip install -r requirements`
- Download data. You can obtain all the benchmarks from [[here](https://drive.google.com/file/d/1fHdkVSgSYG6Um5ChYMv4Dlqp1ZH9BRIq/view?usp=sharing)]. Note that SWaT and WADI datasets are not listed here according to the privacy license of these two datasets. You can go to this [[website](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)] to request these two datasets by filling in the form. 
You can make a dir called `all_datasets`, put the dir into the root path of MADLLM, and download the benchmarks into the dir.


## Train and Test
We provide the experiment scripts of all benchmarks under the folder `./scripts`. 

For example, to run the test on dataset MSL, just run `bash ./scripts/MSL.sh`. You can change the configuration of the experiment in the script. Here are explanation of some parameters:
```bash
--use_skip_embedding     whether use skip embedding
--use_feature_embedding  whether use feature embedding
--use_prompt_pool        whether use prompt pool
--top_k                  the prompt selected in the sub-pool
--prompt_len             the length of each prompt value
--nb_random_samples      the number of negative patches 
--training               whether use pre-trained checkpoints
--few_shot               whether use 20% of the training dataset
```

We use the pre-trained GPT2 model as the LLM model. If you cannot download the model from huggingface website, you can also download it from [[here](https://drive.google.com/file/d/1E93dyt_3MC_3LmMe-_fAxYCC8WTxdq50/view?usp=sharing)]. Then you can make a new dir 'gpt2/' in the root path of MADLLM and put the downloaded model into the 'gpt2/' dir.



## Test from Pre-trained Checkpoints
We also provide the pre-trained checkpoints of MADLLM in [[here](https://drive.google.com/drive/folders/1sGtKB6UScw1i06mG382eiSQLJWSaPaYF?usp=sharing)]. There are two files in this url, where 'SMD.zip' is the checkpoint of MADLLM on SMD dataset and 'checkpoints.zip' is the checkpoint of MADLLM on other five datasets.

Download the two files in the url, unzip them all, make a new dir 'checkpoints/' in the root path of MADLLM and put the downloaded files into the 'checkpoints' dir.

For example, to run the experiment of MSL dataset from checkpoint, just write the configuration `--training = 0` into `./scripts/MSL.sh`. Then write the parameter `--model_name` as the pre-trained checkpoints dir name. Finally run `bash ./scripts/MSL.sh`