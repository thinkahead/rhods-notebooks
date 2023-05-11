#FROM image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow-notebook:latest
FROM tensorflow-notebook:latest

LABEL maintainer="karve@us.ibm.com"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root
RUN apt-get -y -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update
RUN apt-get install -y xvfb ffmpeg cmake gcc zlib1g-dev zlib1g-dev freeglut3-dev python3-pip
RUN conda update -n base -c conda-forge conda
RUN conda install -y jupyterlab-git
COPY customenv.yml customenv.yml
RUN conda env create -f customenv.yml
#RUN pip3 uninstall torch torchvision -y;pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu117
#RUN pip3 install opencv-python transformers diffusers lpips accelerate

USER root

SHELL ["conda","run","-n","customenv","/bin/bash","-c"]
RUN python -m ipykernel install --name customenv --display-name "customenv"

RUN chmod -R 777 /home/jovyan
USER ${NB_UID}
SHELL ["/bin/bash","-c"]
RUN conda init
RUN echo 'conda activate customenv' >> ~/.bashrc
