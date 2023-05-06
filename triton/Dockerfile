#FROM nvcr.io/nvidia/tritonserver:22.09-py3 AS base
FROM nvcr.io/nvidia/tritonserver:23.04-py3 AS base

LABEL maintainer="karve@us.ibm.com"
LABEL project="CEA"
LABEL CEA="DefectDetection"

# some problem from nvidia - remove this later
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential  && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt update
RUN apt install ffmpeg libsm6 libxext6 -y

RUN pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu117

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
#RUN conda install pytorch torchvision cudatoolkit=11 -c pytorch-nightly

WORKDIR /workspace
#COPY certificates /certificates
RUN pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu117

#ENTRYPOINT ["tritonserver", "--model-repository=/models", "--grpc-use-ssl=1", "--grpc-use-ssl-mutual=1", "--grpc-server-cert", "/certificates/server.crt", "--grpc-server-key", "/certificates/server.key", "--grpc-root-cert", "/certificates/ca.crt"]
ENTRYPOINT ["tritonserver", \
            "--model-repository=s3://rook-ceph-rgw-ocs-storagecluster-cephobjectstore.openshift-storage.svc:80/alexei-object-store-65e189ee-454f-455a-a1be-16b00788f3bc/triton-model-repository", \
            "--grpc-use-ssl=1", \
            "--grpc-use-ssl-mutual=1", \
            "--grpc-server-cert=/certificates/server.crt", \
            "--grpc-server-key=/certificates/server.key", \
            "--grpc-root-cert=/certificates/ca.crt", \
            "--grpc-keepalive-time=10000", \
            "--grpc-keepalive-timeout=999999999", \
            "--grpc-keepalive-permit-without-calls=True", \
            "--grpc-http2-max-pings-without-data=0", \
            "--grpc-http2-min-recv-ping-interval-without-data=5000", \
            "--grpc-http2-max-ping-strikes=0"]
# "--model-repository=s3://s3.openshift-storage.svc.cluster.local:80/spark-demo-07ae575f-1b38-4064-b358-c0f7e1a88ddb/triton-model-repository", \
