# NLP 243 Final Project

## Inference

```bash
python inference.py --model_dir semeval25-unlearning-model -c config/config.json
```

Accelerator will automatically choose which device to use (CPU or GPU) based on the availability.

## Training

```bash
accelerate launch --config_file <accelerate confil yaml file> train_run.py -c <config file> -n <experiment name> -p <print frequency>
```
