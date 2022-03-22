# Training with sagemaker

code example to use sagemaker for script-mode training

## Getting started

create an .env file with all your secret variables.

```
AWS_DEFAULT_PROFILE=your-aws-cli-profile
AWS_DEFAULT_REGION=your-aws-region
SAGEMAKER_ROLE_NAME=your-sagemaker-execution-role
```

note:
- to use sagemaker local mode or any sagemaker resources your need an execution role.

install python packages

```
pip install -r requirements.txt
```

run sagemaker local training.

```
python training/execute.py
```

