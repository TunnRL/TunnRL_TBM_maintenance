#!/bin/bash

# installs the poetry packaging system into your system
curl -sSL https://install.python-poetry.org | python3 -

# add poetry to path
echo 'export PATH="$HOME/.poetry/bin:$PATH"' >> ~/.bash_aliases