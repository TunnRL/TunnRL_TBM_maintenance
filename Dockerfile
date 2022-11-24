# Building a docker image in linux include linux programs, git, gnu make etc.
# A 2-stage build has benefits
# STAGE1
#########
# consider slim-bullseye
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
# throws away files from first stage when we start 2nd stage
FROM python:3.10.7-slim as main-stage

# get latest update in linux
# RUN apt-get update

# Install project package and dependencies
COPY --from=requirements-stage /requirements.txt /
COPY --from=requirements-stage /dist /dist
RUN pip install -r requirements.txt --no-deps
RUN pip install --no-deps --no-index dist/*.whl

# copy files that is not a part of the build, scriptfiles and config
RUN mkdir project project/scripts
COPY ./src/*.py /project/scripts
COPY ./src/config /project/scripts/config

WORKDIR /project

RUN ls
CMD "ls"

# CMD ["python", "./src/A_main_hydra.py", "EXP.MODE='training'"]