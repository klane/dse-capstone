ARG BASE_CONTAINER=gitlab-registry.nautilus.optiputer.net/prp/jupyter-stack/prp

FROM $BASE_CONTAINER

COPY requirements.txt ./
COPY cleaned_code.zip ./
COPY traffic_bayArea_station_allStations_12pts.pkl ./

#RUN mkdir -p data 
#COPY train.npz test.npz val.npz adj_mx_new_bay.pkl data/
#RUN git clone https://github.com/akashshah59/DCRNN_PyTorch.git

LABEL maintainer="Torch TS project"

RUN pip install --quiet --no-cache-dir -r requirements.txt

USER root
