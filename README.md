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

### Sagemaker Script Mode

- Bring your own model with sagemaker script mode (BYOM) | https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-with-amazon-sagemaker-script-mode

