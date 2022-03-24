import os
import io
import pandas as pd
import numpy as np
import joblib

def model_fn(model_dir):
    """
    model_fn
        model_dir: (sting) specifies location of saved model
    This function is used by AWS Sagemaker to load the model for deployment.
    It does this by simply loading the model that was saved at the end of the
    __main__ training block above and returning it to be used by the predict_fn
    function below.
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    """
    input_fn
        request_body: the body of the request sent to the model. The type can vary.
        request_content_type: (string) specifies the format/variable type of the request
    This function is used by AWS Sagemaker to format a request body that is sent to
    the deployed model.
    In order to do this, we must transform the request body into a numpy array and
    return that array to be used by the predict_fn function below.
    Note: Oftentimes, you will have multiple cases in order to
    handle various request_content_types. Howver, in this simple case, we are
    only going to accept text/csv and raise an error for all other formats.
    """
    if request_content_type == 'text/csv':
        if type(request_body) == str:
            # Load dataset
            df = pd.read_csv(io.StringIO(request_body), header=None)
        else:
            df = pd.read_csv(io.StringIO(request_body.decode()), header=None)
        return df.values
    else:
        raise ValueError("Thie model only supports text/csv input")

def predict_fn(input_data, model):
    """
    predict_fn
        input_data: (numpy array) returned array from input_fn above
        model (sklearn model) returned model loaded from model_fn above
    This function is used by AWS Sagemaker to make the prediction on the data
    formatted by the input_fn above using the trained model.
    """
    return model.predict(input_data)

def output_fn(prediction, content_type):
    """
    output_fn
        prediction: the returned value from predict_fn above
        content_type: (string) the content type the endpoint expects to be returned
    This function reformats the predictions returned from predict_fn to the final
    format that will be returned as the API call response.
    Note: While we don't use content_type in this example, oftentimes you will use
    that argument to handle different expected return types.
    """

    return {
        'target': prediction.astype(int).tolist()
    }
