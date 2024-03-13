## Edited monai-deploy-app-sdk

This is a fork of monai-deploy-app-sdk at tag v0.5.1. The fork edits [packager/templates.py](https://github.com/tomaroberts/monai-deploy-app-sdk/blob/main/monai/deploy/packager/templates.py)https://github.com/tomaroberts/monai-deploy-app-sdk/blob/main/monai/deploy/packager/templates.py to force the MAP to use Python3.9 instead of default within the MAP BASE_IMAGE. This is often Python3.8 in older base images, which is incompatible with TotalSegmentator.
