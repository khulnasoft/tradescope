FROM khulnasoft/tradescope:latest

USER root
# Install dependencies
COPY requirements-dev.txt /tradescope/

RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog \
    && apt-get -y install --no-install-recommends git sudo vim build-essential \
    && apt-get clean \
    && mkdir -p /home/tsuser/.vscode-server /home/tsuser/.vscode-server-insiders /home/tsuser/commandhistory \
    && echo "export PROMPT_COMMAND='history -a'" >> /home/tsuser/.bashrc \
    && echo "export HISTFILE=~/commandhistory/.bash_history" >> /home/tsuser/.bashrc \
    && chown tsuser:tsuser -R /home/tsuser/.local/ \
    && chown tsuser: -R /home/tsuser/

USER tsuser

RUN pip install --user autopep8 -r docs/requirements-docs.txt -r requirements-dev.txt --no-cache-dir

# Empty the ENTRYPOINT to allow all commands
ENTRYPOINT []
