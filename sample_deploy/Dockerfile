FROM python:3.7.13

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENV PORT=$PORT

ENTRYPOINT ["python"]

CMD ["app.py"]