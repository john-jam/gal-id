FROM tensorflow/tensorflow:2.7.0-gpu

ENV POETRY_PATH=/opt/poetry
ENV PATH="$POETRY_PATH/bin:$PATH"

# Install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -\
    && mv $HOME/.poetry $POETRY_PATH \
    && poetry config virtualenvs.create false

# Define the current workdir
WORKDIR /deploy/apps/giapi

# Install dependencies
COPY apps/giapi/poetry.lock apps/giapi/pyproject.toml ./
COPY libs/gicommon/ /deploy/libs/gicommon/

RUN cd /deploy/apps/giapi/\
    && poetry install -vvv

# Copy the sources
COPY apps/giapi/run.py /deploy/apps/giapi/
COPY apps/giapi/giapi /deploy/apps/giapi/giapi/

# Run our module
ENTRYPOINT ["python", "run.py"]