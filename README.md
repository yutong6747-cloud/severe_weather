# yolo11__WRE_ScOmniAttention_RDSD

This repository keeps only the custom model you asked for.
It is organized so the project can be published on GitHub without other experiment models or unrelated configs.

## Structure

- `ultralytics/nn/modules/custom.py`: all custom blocks and the custom detection head
- `ultralytics/nn/tasks.py`: registration logic for the custom modules
- `cfg/models/yolo11_WRE_ScOmniAttention_RDSD.yaml`: model config
- `cfg/datasets/severe_weather.yaml`: dataset config template
- `severe_weather`:somes datasets
- `train_weather.py`: Training Code


## Run

```bash
pip install -r requirements.txt
python train.py
```

Before training, update `cfg/datasets/weather_weather.yaml` to your dataset paths.

