# Docker file to build an image that train a ML-model
#####################################################

# Build the image in 2 stages
# - reduce size of image
# - only install dependencies and new package in image, no poetry etc.
# - safer
# - don't expose code in package

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

# WORKDIR /project #try this for better organization in container

# Directories for optional mounting of external directories.
# In singularity it is perhaps the best option to mount a superdirectory with these
#directories into the directory that is the cwd, eg. /home/tfh/projects/tbm-rl
RUN mkdir scripts optimization checkpoints results graphics experiments
# copy files that is not a part of the build, scriptfiles and config
COPY ./src/*.py /scripts
COPY ./src/config /scripts/config


CMD ["python", "./scripts/A_main_hydra.py", "EXP.MODE='training'"]
