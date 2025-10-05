# Variables
PROJECT_ID="your-gcp-project"
REGION="us-central1"
IMAGE_URI="us-central1-docker.pkg.dev/$PROJECT_ID/vertex-ai-dask/train:latest"

# Submit custom training job with 4 workers
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=dask-optuna-job \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=4,container-image-uri=$IMAGE_URI
