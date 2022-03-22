import os
import argparse
from sagemaker.sklearn.estimator import SKLearn

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run training job')
    parser.add_argument('--local', default=True, action='store_false')
    parser.add_argument('--inputs-provided', default=False, action='store_false')
    parser.add_argument('--input_path', default=None, type=str)
    parser.add_argument('--output_path', default="outputs", type=str, help="path to store model")
    parser.add_argument('--hyperparameters', '-p', nargs=2, default=[], action='append')
    args = parser.parse_args()

    role = os.environ["SAGEMAKER_ROLE_NAME"]

    hyperparameters = {"max_depth": 10, "n_jobs": 4, "n_estimators": 120}

    for p in args.hyperparameters:
        hyperparameters.update({p[0]: p[1]})

    print(hyperparameters)

    if args.local:
        train_instance_type = "local"
        inputs = {"train": f"file://{args.input_path}/train", "test": f"file://{args.input_path}/test"}
        output_path = f"file://{args.output_path}"
    else:
        train_instance_type = "ml.c5.xlarge"
        inputs = {"train": f"{args.input_path}/train", "test": f"{args.input_path}/test"}
        output_path = f"{args.output_path}"

    estimator_parameters = {
        "entry_point": "train.py", # script name to run main
        "source_dir": "training/code", # path to where scripts are stored locally
        "framework_version": "0.23-1", # version of sklearn framework to use
        "py_version": "py3", # version of python to use
        "instance_type": train_instance_type,
        "instance_count": 1,
        "hyperparameters": hyperparameters,
        "output_path": output_path,
        "dependencies": [], # provide python packages to install
        "role": role,
        "base_job_name": "randomforestregressor-model"  # give sagemaker a naming seed, for which it generates a unique name
    }

    estimator = SKLearn(**estimator_parameters)

    if args.inputs_provided:
        estimator.fit(inputs=inputs)
    else:
        estimator.fit()
