FROM python:3.8.10-slim

RUN apt-get update && apt-get install -y \
  curl \
  && rm -rf /var/lib/apt/lists/*

ENV POETRY_PATH=/opt/poetry
ENV PATH="$POETRY_PATH/bin:$PATH"

# Install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -\
    && mv $HOME/.poetry $POETRY_PATH \
    && poetry config virtualenvs.create false

# Define the current workdir
WORKDIR /deploy/apps/giwebapp

# Install dependencies
COPY apps/giwebapp/poetry.lock apps/giwebapp/pyproject.toml ./
COPY libs/gicommon/ /deploy/libs/gicommon/

RUN cd /deploy/apps/giwebapp/\
    && poetry install -vvv

# Copy the sources
COPY apps/giwebapp/run.py /deploy/apps/giwebapp/
COPY apps/giwebapp/assets /deploy/apps/giwebapp/assets/
COPY apps/giwebapp/.streamlit /deploy/apps/giwebapp/.streamlit/
COPY apps/giwebapp/giwebapp /deploy/apps/giwebapp/giwebapp/

# Run our module
ENTRYPOINT ["streamlit", "run", "run.py"]