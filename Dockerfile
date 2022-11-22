FROM python:3.10.7-slim as requirements-stage

RUN pip install poetry==1.2.2

# and only reinstall if they are changed
COPY poetry.lock / 
COPY pyproject.toml /
RUN poetry export -f requirements.txt -o requirements.txt

# add source code into the image
COPY ./src /src
COPY README.md /
# RUN mkdir experiments graphics optimization results checkpoints
# RUN mkdir experiments/mlruns experiments/hydra_outputs

# build
RUN poetry build

# throws away files from first stage when we start 2nd stage
FROM python:3.10.7-slim

# Install project package and dependencies
COPY --from=requirements-stage /requirements.txt /
COPY --from=requirements-stage /dist /dist
RUN pip install -r requirements.txt --no-deps
RUN pip install --no-deps --no-index dist/*.whl
# copy files that is not a part of the build
COPY ./src/A_main_hydra.py /src/A_main_hydra.py
COPY ./src/B_optimization_analyzer.py /src/B_optimization_analyzer.py
COPY ./src/config /src/config

RUN ls
CMD "ls"

# CMD ["python", "./src/A_main_hydra.py", "EXP.MODE='training'"]
