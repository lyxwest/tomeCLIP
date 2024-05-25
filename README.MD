# Token merge for VideoCLIP
This is the repository for migrating ToMe to videoclip work. For more information about ToMe, please refer to ``ToMe_README.MD``.

## Install

1. Clone CLIP repository
```bash
pip install git+https://github.com/openai/CLIP.git
```

2. Requirements

Please refer to ``requirements.txt``


## Usage

### 1. Throughput benchmark

See `examples/validation_videoclip.ipynb`


### 2. Infrence 

We use HMDB51 video dataset.

```bash
bash script/run_test.sh configs/hmdb51/hmdb_few_shot.yaml <PATH_TO_MODEL>
```