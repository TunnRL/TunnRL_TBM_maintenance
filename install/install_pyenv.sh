#!/bin/bash

# Installs pyenv for controlling python versions in your linux system
# More info: https://realpython.com/intro-to-pyenv/
# More info about the linux configuration system: https://www.baeldung.com/linux/bashrc-vs-bash-profile-vs-profile

# Make sure you have the right build dependencies for install pyenv
# Otherwise you will get warnings in the installation and later errors in use
# Here you also get curl, which you will use for the actual installation.
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

# install pyenv
curl https://pyenv.run | bash

# add pyenv to path in your bash config file
echo '' >> ~/.bash_aliases
echo '####################################################################'  >> ~/.bash_aliases
echo '# pyenv                                                            #'  >> ~/.bash_aliases
echo '####################################################################'  >> ~/.bash_aliases
echo '' >> ~/.bash_aliases
echo 'export PATH="$HOME/.pyenv/bin:$PATH"'  >> ~/.bash_aliases
echo 'eval "$(pyenv init --path)"'  >> ~/.bash_aliases
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_aliases

source ~/.bash_aliases