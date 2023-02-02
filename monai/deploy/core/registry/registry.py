
from abc import ABC, abstractmethod
from typing import Any, Hashable, KeysView, Optional


class Registry(ABC):
    """Base class for model registry"""

    @abstractmethod
    def get_model(self, model_uri: str) -> Any:
        """
        Get model from model registry.

        Args:
            model_uri (str): URI of requested model in registry

        Returns:
            value (Any): A path to the downloaded model.
        """
        pass

    @abstractmethod
    def validate_registry(self) -> bool:
        """
        Returns true is registry is alive
        :return: bool
        """
        pass

    @abstractmethod
    def validate_model(self, model_path):
        """
        Validate that model is compatible with monai-deploy-app-sdk
        :param model_path:
        :return:
        """
