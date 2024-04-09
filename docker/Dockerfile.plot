ARG sourceimage=tradescopeorg/tradescope
ARG sourcetag=develop
FROM ${sourceimage}:${sourcetag}

# Install dependencies
COPY requirements-plot.txt /tradescope/

RUN pip install -r requirements-plot.txt --user --no-cache-dir
