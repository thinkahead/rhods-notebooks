FROM ghcr.io/foundation-model-stack/base:pytorch-latest-nightly-20230426
#FROM quay.io/michaelclifford/mnist-test:latest
COPY mnist_fashion.py mnist_fashion.py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pwd
RUN whoami
RUN chmod -R 777 .
#RUN chmod -R 777 /app
RUN chmod -R 777 /workspace
