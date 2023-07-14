FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:latest
RUN pip install sconf datasets zss sentencepiece tensorboard transformers==4.25.1 timm==0.5.4 pytorch-lightning==1.9.0
RUN pip uninstall nvidia-cublas-cu11 -y
RUN pip install nvidia-cublas-cu12 nltk google-cloud-aiplatform

WORKDIR /

COPY train.py /src/train.py
COPY donut /src/donut
COPY config /src/config

ENTRYPOINT ["python", "-m", "src.train", "--config", "/src/config/train_cord.yaml", "--exp_version", "test_experiment"]