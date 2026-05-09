#!/usr/bin/env bash
# Manual one-shot deploy. Run from the finagent/ directory.
# For CI/CD use cloudbuild.yaml instead.
set -euo pipefail

PROJECT=aibuild-495014
REGION=us-central1
SERVICE=finagent
SA=finagent-sa@aibuild-495014.iam.gserviceaccount.com
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${SERVICE}/${SERVICE}"
TAG=$(git rev-parse --short HEAD)

echo "==> Authenticating Docker to Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "==> Building image: ${IMAGE}:${TAG}"
docker build \
  --file deploy/Dockerfile \
  --tag "${IMAGE}:${TAG}" \
  --tag "${IMAGE}:latest" \
  .

echo "==> Pushing image"
docker push "${IMAGE}:${TAG}"
docker push "${IMAGE}:latest"

echo "==> Deploying to Cloud Run"
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}:${TAG}" \
  --region="${REGION}" \
  --project="${PROJECT}" \
  --service-account="${SA}" \
  --port=8080 \
  --memory=4Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=10 \
  --concurrency=4 \
  --timeout=300 \
  --set-secrets="\
GOOGLE_API_KEY=GOOGLE_API_KEY:latest,\
GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT:latest,\
GOOGLE_CLOUD_LOCATION=GOOGLE_CLOUD_LOCATION:latest,\
GOOGLE_GENAI_USE_VERTEXAI=GOOGLE_GENAI_USE_VERTEXAI:latest,\
MODEL_ORCHESTRATOR=MODEL_ORCHESTRATOR:latest,\
MODEL_ANALYST=MODEL_ANALYST:latest,\
GCS_BUCKET_URI=GCS_BUCKET_URI:latest,\
VECTOR_INDEX_ID=VECTOR_INDEX_ID:latest,\
VECTOR_INDEX_ENDPOINT_ID=VECTOR_INDEX_ENDPOINT_ID:latest,\
DEPLOYED_INDEX_ID=DEPLOYED_INDEX_ID:latest,\
EMBEDDING_MODEL=EMBEDDING_MODEL:latest,\
GENERATION_MODEL=GENERATION_MODEL:latest,\
EMBEDDING_DIMENSIONS=EMBEDDING_DIMENSIONS:latest,\
TOP_K=TOP_K:latest,\
RRF_RANKING_ALPHA=RRF_RANKING_ALPHA:latest,\
EMBEDDING_BATCH_SIZE=EMBEDDING_BATCH_SIZE:latest,\
API_MAX_RETRIES=API_MAX_RETRIES:latest,\
RISK_FREE_RATE=RISK_FREE_RATE:latest,\
BENCHMARK_TICKER=BENCHMARK_TICKER:latest,\
DOCUMENTS_DIR=DOCUMENTS_DIR:latest,\
DATA_DIR=DATA_DIR:latest" \
  --no-allow-unauthenticated \
  --platform=managed

SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
  --region="${REGION}" --project="${PROJECT}" \
  --format='value(status.url)')

echo ""
echo "==> Deployed: ${SERVICE_URL}"
echo ""
echo "==> Verify:"
echo "    TOKEN=\$(gcloud auth print-identity-token)"
echo "    curl -H \"Authorization: Bearer \$TOKEN\" ${SERVICE_URL}/health"
