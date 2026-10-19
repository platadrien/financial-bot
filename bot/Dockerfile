FROM python:3.13-slim-buster

WORKDIR /python-docker

COPY . .
RUN pip install --timeout=300 -r requirements.txt

CMD [ "python", "bot.py"]