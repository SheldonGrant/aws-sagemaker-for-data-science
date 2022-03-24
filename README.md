# aws-sagemaker-for-data-science
a simplified tutorial for quickly interacting and understanding the core features of sagemaker for data science

## Understanding sagemaker

#### Overview

In sagemaker there are 4 levels:
1. [sagemaker API layer](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job) this is effectively boto3 layer which is leveraged by python sdk.
2. [sagemaker estimator layer](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) built on top of the boto3 API. And supports any docker image (BYOC - bring your own docker container)
3. [sagemaker framework layer](https://sagemaker.readthedocs.io/en/stable/frameworks/index.html) built on estimator layer. Here sagemaker does the heavy lifting, building docker images baseed on popular framework like SKLEARN or PYTORCH, etc. This is where script mode integrates.
4. [sagemaker built in algorithm layer](https://sagemaker.readthedocs.io/en/stable/algorithms/index.html) built on top framework layer and script mode. These are the sagemaker 1st party algorithms. You only provide the data, and sagemaker does everything else.

#### Suggestions

- start at script mode or framework level
- use 1st party algrorithms if possible, as it removes all the headache of writing your own scripts and often provides optimized models for simple model varieties i.e. classification, regression, clustering
- use estimator layer if you need to provide your own docker containers for custom frameworks

#### Docker, Sagemaker configuration and Environment variables

At it's core, sagemaker is a configuration system built on docker and docker-compose. It passes your sdk and API configuration to the docker runtime as environment variables at runtime.

To get a grasp of where you can find your data and model artifacts, take time to examine the following:
- all data is written to or picked at a path starting with `/opt/ml`
- Input, Output and Model paths consist of a base path called a channel, where the channel name matches the name of the data passed i.e. `/opt/ml/input/data/{channel_name}`.
- your data will be fetched from an s3 directory prefix and stored at the matching channel_name i.e `/opt/ml/input/data/{channel_name}/data.csv`. 
- pretrained or checkpointed models are written at `/opt/ml/model/{channel}`
- write your trained model artifacts to `/opt/ml/model/{channel}/{file.name}`
- write your other output artifacts i.e. graphics or lookup tables, etc to `/opt/ml/output/{channel}/{file.name}`
- all output channels and model channels are written back to s3 automatically by sagemaker according to the output_data_path in the Estimator config.

##### Environment Variables

Sagemaker makes use of environment variables to provide base paths to input data, output data and models

- important-environment-variables  | https://github.com/aws/sagemaker-containers#important-environment-variables

### Sagemaker Script Mode

- Bring your own model with sagemaker script mode (BYOM) | https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-with-amazon-sagemaker-script-mode

