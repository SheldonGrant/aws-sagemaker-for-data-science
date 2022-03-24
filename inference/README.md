# Deploy for inference

A few key concepts in sagemaker inference include:

| **concept** | **definition** |
|---|---|
| model artifacts | .joblib or .pkl type object(s) stored in s3 which are written after a training job. These are loaded in to the model resource on creation. |
| sagemaker model resource | an AWS sagemaker infrastructural resource which combines model artifacts and model environment settings |
| sagemaker endpoint | an AWS sagemaker infrastructural resource which receives requests and returns responses |
| sagemaker endpoint configuration | a configuration resource in sagemaker, which describes an endpoint. Specifically, which model and how a model will be served there. |

## Workflow for deploying a model

1. create a sagemaker model resource
2. create a sagemaker endpoint configuration
3. create / update a saegmaker endpoint with a configuration

note:
- these steps are abstracted in the sagemaker python sdk when you call deploy it creates a configuration and updates the endpoint.

## Testing Model serving

Deploy your model. In local mode, you need to provide your path to model.tar.gz files. In cloud, we reference the training job.

### Local

deploy model
```
python inference/deploy.py --model-data-path=<path/to/model/artifacts/model.tar.gz>
```

```
curl -XPOST "http://localhost:8080/invocations" -d @inference/payload.target2.csv -H "Content-Type: text/csv"
```

### Cloud

deploy model
```
python inference/deploy.py --training-job-name=<your/training/job/name> --cloud
```

query model

```
python inference/invoke.py --file-path=inference/payload.target2.csv
```

note:
- cloud mode, creates a sagemaker endpoint. This takes time, a few minutes.
- remember to delete your sagemaker endpoint and endpoint configuration from the console once you are done.
