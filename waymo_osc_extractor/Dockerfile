FROM registry.gitlab.iavgroup.local/caledon/scenariobased-testing/waymo_osc_extractor:builder-1298301

WORKDIR /app

COPY requirements*.txt ./

COPY certs ./certs/
COPY waymo_osc_extractor ./waymo_osc_extractor/
COPY features/ ./features/
COPY steps/ ./steps/
COPY kube_runner.py ./kube_runner.py

RUN python -m pip --disable-pip-version-check install --no-cache-dir --index-url=https://artifactory.iav.com/artifactory/api/pypi/pypi-remote/simple -r requirements.txt
RUN python -m pip --disable-pip-version-check install --no-cache-dir --index-url=https://artifactory.iav.com/artifactory/api/pypi/pypi-remote/simple -r waymo_osc_extractor/waymo_scenario_tools/requirements.txt

CMD [ "/bin/bash" ]
