import os
import argparse
from sagemaker.estimator import Estimator
from sagemaker.sklearn.model import SKLearnModel

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='deploy inference job')
    parser.add_argument('--cloud', default=False, action='store_true')
    parser.add_argument('--training-job-name', default=None, type=str)
    parser.add_argument('--model-data-path', default=None, type=str)
    args = parser.parse_args()

    role = os.environ["SAGEMAKER_ROLE_NAME"]
    developer_name = os.environ.get("AWS_TAG_DEVELOPER_NAME" "datascience-test")

    if args.cloud:
        # fetch name of previously run training job
        estimator = Estimator.attach(
            args.training_job_name,
            sagemaker_session=None,
            model_channel_name='model'
        )
        train_instance_type = "ml.m5.large"
        model_data = estimator.model_data
        model_name = args.training_job_name
    else:
        train_instance_type = "local"
        model_data = args.model_data_path
        model_name = "my-test-model"

    # create a model
    create_model_configuration = {
        "name": model_name,
        "model_data": model_data,
        "framework_version": "0.23-1",  # version of sklearn framework to use
        "py_version": "py3",  # version of python to use
        "entry_point": "inference.py",  # script name to run main
        "source_dir": "inference/code",  # path to where scripts are stored locally
        "role": role,
        "dependencies": ["requirements.txt"], # provide files needed i.e. utils.py or requirements.txt, etc
        "env": {
            'SAGEMAKER_REQUIREMENTS': 'requirements.txt', # file to pip install -r requirements.txt
        }
    }

    # use SKLearnModel framework to get the defaults for image_uri, etc
    model = SKLearnModel(
        **create_model_configuration
    )
    
    # deploy
    model.deploy(
        initial_instance_count=1,
        instance_type=train_instance_type,
        # update_endpoint=True,
        endpoint_name='existing-endpoint'
    )
