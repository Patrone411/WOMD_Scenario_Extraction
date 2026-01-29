FROM docker.artifactory.iav.com/python:3.8.20

RUN python -m pip --disable-pip-version-check install boto3 geopandas waymo-open-dataset-tf-2-12-0 --index-url=https://artifactory.iav.com/artifactory/api/pypi/pypi-remote/simple

CMD [ "/bin/bash" ]
