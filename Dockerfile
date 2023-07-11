FROM --platform=linux/x86_64 ubuntu AS base
FROM base
COPY . /app
WORKDIR /app
RUN apt update && \
    apt install -y python3 python3-pip &&\
    pip install tensorflow==2.11.* tensorflow-text==2.11.* pandas numpy

CMD python3 inference.py

