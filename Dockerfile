# eventually use python:3.10.7 or 3.10.5-slim
FROM python:3.10.5-slim

WORKDIR /project
# More info: https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
# install the latest version of poetry
# If you want a specific python version you need to use pip
# ie. RUN pip install poetry==1.2
# RUN curl -sSL https://install.python-poetry.org | python3 -
# add poetry to path

# ENV PATH="$HOME/.poetry/bin:$PATH"

RUN pip install poetry==1.1.14

# make sure you don't creante another virtual env, in addition to the docker env
RUN poetry config virtualenvs.create false

# Copy requirement files to workdir. This will cache our requirements
# and only reinstall if they are changed
COPY poetry.lock pyproject.toml ./

# add source code into the image
COPY . ./

# install dependencies from poetry.lock
RUN poetry install

# list project structure
RUN ls

# activate the environment when starting the container
CMD ["poetry", "shell"]

# it should now be possible to run scripts on the container
