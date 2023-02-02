from typing import Dict, Optional

from monai.deploy.exceptions import UnknownTypeError
from .mlflow_registry import MLflowRegistry
from .registry import Registry


class RegistryFactory:
    """RegistryFactory is an abstract class that provides a way to create a registry object."""

    @staticmethod
    def create(registry_type: str, registry_params: Optional[Dict] = None) -> Registry:
        """Creates a registry object.

        Args:
            registry_type (str): A type of the registry.
            registry_params (Dict): A dictionary of parameters of the registry.

        Returns:
            Registry: A model registry object.
        """

        if registry_type == "mlflow":
            return MLflowRegistry(**registry_params)
        else:
            raise UnknownTypeError(f"Unknown registry type: {registry_type}")
