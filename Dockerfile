# Docker file to build an image that train a ML-model
#####################################################

# Build the image in 2 stages
# - reduce size of image
# - only install dependencies and own program in image, no poetry etc.
# - safer

# STAGE1
#########
FROM python:3.10.7-slim as requirements-stage

RUN pip install poetry==1.2.2

# and only reinstall if they are changed
COPY poetry.lock / 
COPY pyproject.toml /
RUN poetry export -f requirements.txt -o requirements.txt

# add source code into the image
COPY ./src /src
COPY README.md /

# build
RUN poetry build

# STAGE2
#########
# Except for requirements and the python packakge we throw away files from first stage 
# when we start 2nd stage
FROM python:3.10.7-slim as main-stage

# get latest update of linux
RUN apt-get -y update && apt-get upgrade -y

# Install project package and dependencies
COPY --from=requirements-stage /requirements.txt /
COPY --from=requirements-stage /dist /dist
RUN pip install -r requirements.txt --no-deps
RUN pip install --no-deps --no-index dist/*.whl

# copy files that is not a part of the build, scriptfiles and config
# RUN mkdir project project/scripts project/exp_results
RUN mkdir scripts, exp_results, data
# COPY ./src/*.py /project/scripts
COPY ./src/*.py /scripts
# COPY ./src/config /project/scripts/config
COPY ./src/config /scripts/config

# WORKDIR /project

CMD ["python", "./scripts/A_main_hydra.py", "EXP.MODE='training'"]
