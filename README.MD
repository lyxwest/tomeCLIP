# Token merge for VideoCLIP
This is the repository for migrating ToMe to videoclip work. For more information about ToMe, please refer to ``ToMe_README.MD``.

## Install

1. Clone CLIP repository
```bash
pip install git+https://github.com/openai/CLIP.git
```

2. Requirements

Please refer to ``requirements.txt``


3. Data Preparation

We use HMDB51 video dataset and extract videos into frames for fast reading. You can refer to [MVF](https://github.com/whwu95/MVFNet/blob/main/data_process/DATASETS.md) for the detaied guide of data processing.


## Usage

### 1. Throughput benchmark

See `examples/validation_videoclip.ipynb`

This process does not require a trained model and loading a dataset. The principle is to :
(1) build the model,   
(2) randomly generate tensors of the corresponding data size,  
(3) input them into the model to calculate throughput.

### 2. Training

``` bash
bash script/run_train.sh configs/hmdb51/hmdb_few_shot.yaml
```
Since we are currently optimizing only for inference, you can freely train a model on the HMDB51 dataset (or use the checkpoint we will provide later). This won't take much of your time; training on 4 2080ti GPUs takes 1 hour and 10 minutes.


### 3. Infrence 

```bash
bash script/run_test.sh configs/hmdb51/hmdb_few_shot.yaml <PATH_TO_MODEL>
```