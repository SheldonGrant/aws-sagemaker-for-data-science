import boto3
import io
import json
import argparse
import pandas as pd

def encode_body(data):
    csv_file = io.StringIO()
    data.to_csv(csv_file, sep=",", header=False, index=False)
    return csv_file.getvalue()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run training job')
    parser.add_argument('--file-path', default=None, type=str)
    args = parser.parse_args()

    sagemaker_runtime = boto3.client('sagemaker-runtime')

    data = pd.read_csv(args.file_path, header=None)

    request_body = encode_body(data)
    print(request_body)

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName="existing-endpoint",
        Body=request_body,
        ContentType='text/csv',
    )

    print("Response from sagemaker endpoint")
    result = json.loads(response['Body'].read().decode())

    print(result)
