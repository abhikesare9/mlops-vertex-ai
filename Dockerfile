From python:3.10-bullseye
WORKDIR /app
USER 0
COPY requirements.txt /app
COPY main.py /app
COPY *.json /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3","main.py"]
EXPOSE 5000
