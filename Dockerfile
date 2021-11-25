FROM tensorflow/tensorflow
WORKDIR /code

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1

# Install pipenv and compilation dependencies
RUN pip install pipenv
RUN apt-get update && apt-get install -y gcc python3-opencv

# Install python dependencies in /.venv
COPY Pipfile .
#COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy --skip-lock

ENV PATH="/code/.venv/bin:$PATH"

COPY ./src /code/app
COPY ./data/model/memento.h5 /code/data/model/memento.h5
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]