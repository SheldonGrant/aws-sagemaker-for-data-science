import os
import argparse
import boto3
from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run training job")
    parser.add_argument("--cloud", default=False, action="store_true")
    parser.add_argument("--inputs-provided", default=False, action="store_true")
    parser.add_argument("--input-path", default=None, type=str)
    parser.add_argument(
        "--output-path",
        default="outputs",
        type=str,
        help="path to store model and outputs",
    )
    parser.add_argument("--hyperparameters", "-p", nargs=2, default=[], action="append")
    args = parser.parse_args()

    role = os.environ["SAGEMAKER_ROLE_NAME"]
    developer_name = os.environ.get("AWS_TAG_DEVELOPER_NAME" "datascience-test")

    hyperparameters = {"max_depth": 10, "n_jobs": 4, "n_estimators": 120}

    for p in args.hyperparameters:
        hyperparameters.update({p[0]: p[1]})

    print(f"Hyperparameters = {hyperparameters}")

    BOTO_SESSION = boto3.Session()
    SM_SESSION = Session(default_bucket=os.environ["SAGEMAKER_DEFAULT_S3_BUCKET"], boto_session=BOTO_SESSION)

    if args.cloud:
        train_instance_type = "ml.m5.large"
        inputs = {
            "train": f"{args.input_path}/train",
            "test": f"{args.input_path}/test",
        }
        output_path = f"{args.output_path}"
    else:
        train_instance_type = "local"
        inputs = {
            "train": f"file://{args.input_path}/train",
            "test": f"file://{args.input_path}/test",
        }
        output_path = f"file://{args.output_path}"

    estimator_parameters = {
        "entry_point": "train.py",  # script name to run main
        "source_dir": "training/code",  # path to where scripts are stored locally
        "framework_version": "0.23-1",  # version of sklearn framework to use
        "py_version": "py3",  # version of python to use
        "instance_type": train_instance_type,
        "instance_count": 1,
        "hyperparameters": hyperparameters,
        "output_path": output_path,
        "model_channel_name": "model",
        "dependencies": [],  # provide python packages to install
        "role": role,
        "base_job_name": "svclassifier-model",  # give sagemaker a naming seed, for which it generates a unique name
        "tags": [{"Key": "developer", "Value": f"{developer_name}"}],
    }

    if args.inputs_provided:
        estimator_parameters.update({"entry_point": "train_provide_data.py"})
        estimator = SKLearn(**estimator_parameters)
        estimator.fit(inputs=inputs)
    else:
        estimator = SKLearn(**estimator_parameters)
        estimator.fit()
