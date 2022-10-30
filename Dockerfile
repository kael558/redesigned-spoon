# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/kael558/redesigned-spoon.git .

COPY .env /app/

RUN pip3 install -r requirements.txt

RUN [ "python", "-c", "import nltk; nltk.download('stopwords'); nltk.download('punkt')"]

ENTRYPOINT ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
