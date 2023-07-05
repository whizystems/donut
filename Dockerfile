# https://cloud.google.com/vertex-ai/docs/training/create-custom-container
FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest

# Copy setup.py, train.py and lightning_module to the container image
COPY setup.py /src/setup.py
COPY train.py /src/train.py
COPY lightning_module.py /src/lightning_module.py
COPY config/train_cord.yaml /src/config/train_cord.yaml

# Debug container structure
#RUN ls -la -R /src

ENV PYTHONUNBUFFERED=0
WORKDIR /src
CMD [ "python", "-u", "train.py"]