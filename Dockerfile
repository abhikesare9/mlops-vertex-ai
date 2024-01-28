From python:3.10-bullseye
WORKDIR /app
USER 0
COPY requirements.txt /app
COPY main.py /app
COPY countVectorizer.pkl /app
COPY model_xgb.pkl /app
COPY scaler.pkl /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3","main.py"]
EXPOSE 5000
