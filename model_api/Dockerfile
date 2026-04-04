FROM python:3.11-slim-buster

WORKDIR /python-docker

COPY ../tools/ ./tools/
COPY . .
RUN pip install  -r model_api/requirements.txt

CMD [ "python", "-m" , "flask", "--app", "model_api/api_model.py", "run", "--host=0.0.0.0"]