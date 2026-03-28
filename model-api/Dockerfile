FROM python:3.11-slim-buster

WORKDIR /python-docker

COPY . .
RUN pip install  -r requirements.txt


CMD [ "python", "-m" , "flask", "--app", "api_model.py", "run", "--host=0.0.0.0"]