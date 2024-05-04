## MADLLM: Multivariate Anomaly Detection for Cloud Systems via Pre-trained LLMs


## Get Start
- Create a new environment using conda. `conda create -n madllm python=3.8`
- Activate the environment. `conda activate madllm`
- Install the required packages. `pip install -r requirements`
- Download data. You can obtain all the benchmarks from [[here](https://drive.google.com/drive/folders/1dYba8m2W0LWfh6btAjdOphi_dWlbd9lB?usp=sharing)]. Note that SWaT and WADI datasets are not listed here according to the privacy license of these two datasets. You can go to this [[website](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)] to request these two datasets by filling in the form. 
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

We use the pre-trained GPT2 model as the LLM model. If you cannot download the model from huggingface website, you can also download it from [[here](https://drive.google.com/drive/folders/1OqzFJ7iIOOObMtJ4eiv60JrgXx0Riqzr?usp=sharing)] and put it into the root path of MADLLM.



## Test from Pre-trained Checkpoints
We also provide the pre-trained checkpoints of MADLLM in [[here](https://drive.google.com/file/d/1CgjZ5tlAwKrgW0bow168tK3iqyzJ_wV6/view?usp=sharing)]. Note that the pre-trained SMD checkpoints is not here because it is a too-large file that our google drive does not have enough space. However, you can train it easily with the bash file we support.

Download the file, unzip it and put it into the './checkpoints/' dir.

For example, to run the experiment of MSL dataset from checkpoint, just write the configuration `--training = 0` into `./scripts/MSL.sh`. Then write the parameter `--model_name` as the pre-trained checkpoints dir name. Finally run `bash ./scripts/MSL.sh`