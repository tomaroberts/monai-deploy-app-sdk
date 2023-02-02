
"""
0. validate mlflow
    can be running as server "MLFLOW_TRACKING_URI', or in local folder?
1. GET model download uri
2. download uri
3. validate
    1. is it a torchscript model?

"""
from typing import Dict, Optional

import os
from urllib.parse import urljoin

import mlflow
import requests

from .registry import Registry


class MLflowRegistry(Registry):
    """MLflow model registry."""

    def __init__(self,  **kwargs: Dict):
        self.base_url = urljoin(self.mlflow_url, 'api/2.0/mlflow/')

    def get_model(self, model_uri):
        return mlflow.artifacts.download_artifacts(model_uri)

    def validate_model(self):
        pass

    def validate_registry(self):
        if os.environ.get('MLFLOW_TRACKING_URI') is not None:
            self.mlflow_url = os.environ['MLFLOW_TRACKING_URI']
            assert self._get_request('experiments/get', {'experiment_id': '1'})  # check server is alive
        else:
            raise Exception('MLFLOW_TRACKING_URI undefined')

    def _get_request(self, endpoint: str, request_params: dict) -> dict:
        url = urljoin(self.base_url, endpoint)
        try:
            r = requests.get(url, params=request_params)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as err:
            raise Exception(err)
