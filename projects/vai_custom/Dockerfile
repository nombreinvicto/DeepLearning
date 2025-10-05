FROM python:3.10

RUN pip install scikit-learn dask-ml dask[distributed] optuna google-cloud-aiplatform

COPY train.py /train.py
ENTRYPOINT ["python", "train.py"]
