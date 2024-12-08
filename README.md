# NLP 243 Final Project

## Fetching Data

Before start training, you need to fetch the data using script `fecth_data.py`. The script will download the dataset and
7B and 1B finetuned model.

## Training

```bash
accelerate launch --config_file <accelerate confil yaml file> train_run.py -c <config file> -n <experiment name> -p <print frequency>
```

Accelerator will automatically choose which device to use (CPU or GPU) based on the availability.

### Examples

#### Gradient Ascent (GA) Method

```bash
accelerate launch --config_file accelerate_single_gpu.yaml train_run.py -c config/config_1B_gradient_ascent.json -n GA_exp -p 10
```

#### Gradient Adjustment Method

```bash
accelerate launch --config_file accelerate_single_gpu.yaml train_run.py -c config/config_1B_gradient_adjustment.json -n gradient_adjustment_exp -p 10
```

Checkpoint directory `chkpt/<experiment name>` and logs directory `logs/<experiment name>` will be automatically
created where the saved models and logs will be stored.

## Inference

```bash
python inference.py --model_dir <stored model directory> -c config/<config file>
```


