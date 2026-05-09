# Deploy

Deploys finagent to **Cloud Run** backed by Vertex AI services (LLM, embeddings, Vector Search).

## Prerequisites (run once)

```bash
# Enable APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com aiplatform.googleapis.com \
  storage.googleapis.com iam.googleapis.com --project=aibuild-495014

# Artifact Registry
gcloud artifacts repositories create finagent \
  --repository-format=docker --location=us-central1 --project=aibuild-495014

# Service account + IAM
gcloud iam service-accounts create finagent-sa \
  --display-name="Finagent Cloud Run SA" --project=aibuild-495014

SA=finagent-sa@aibuild-495014.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding aibuild-495014 \
  --member="serviceAccount:$SA" --role="roles/aiplatform.user"
gcloud storage buckets add-iam-policy-binding gs://finragbucket \
  --member="serviceAccount:$SA" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding aibuild-495014 \
  --member="serviceAccount:$SA" --role="roles/secretmanager.secretAccessor"
```

## Load Secrets

Create one Secret Manager secret per env var. Key values to set correctly:

| Secret | Value |
|---|---|
| `GOOGLE_GENAI_USE_VERTEXAI` | `1` |
| `DOCUMENTS_DIR` | `/app/rag_tool/documents` |
| `DATA_DIR` | `/app/rag_tool/data` |
| `GOOGLE_API_KEY` | your API key (still needed for embeddings) |

```bash
# Create all secret shells
for S in GOOGLE_API_KEY GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_LOCATION \
  GOOGLE_GENAI_USE_VERTEXAI MODEL_ORCHESTRATOR MODEL_ANALYST GCS_BUCKET_URI \
  VECTOR_INDEX_ID VECTOR_INDEX_ENDPOINT_ID DEPLOYED_INDEX_ID EMBEDDING_MODEL \
  GENERATION_MODEL EMBEDDING_DIMENSIONS TOP_K RRF_RANKING_ALPHA EMBEDDING_BATCH_SIZE \
  API_MAX_RETRIES BIGQUERY_DATASET RAG_CORPUS_NAME RISK_FREE_RATE BENCHMARK_TICKER \
  DOCUMENTS_DIR DATA_DIR; do
    gcloud secrets create $S --project=aibuild-495014 2>/dev/null || true
done

# Set each value:
echo -n "VALUE" | gcloud secrets versions add SECRET_NAME --data-file=- --project=aibuild-495014
```

## Deploy

**Option A — Cloud Build (no local Docker needed):**
```bash
# Run from finagent/ directory
gcloud builds submit --config=deploy/cloudbuild.yaml --project=aibuild-495014 .
```

**Option B — Local Docker:**
```bash
bash deploy/deploy.sh
```

## Verify

```bash
URL=$(gcloud run services describe finagent --region=us-central1 \
      --project=aibuild-495014 --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" "$URL/health"
curl -X POST "$URL/chat" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello","user_id":"test"}'
```

## Cloud Run Configuration

| Setting | Value | Reason |
|---|---|---|
| Memory | 4 GiB | 6 parallel sub-agents + pandas/numpy/scipy |
| CPU | 2 | asyncio thread pool for yfinance |
| Concurrency | 4 | ~1 GiB per concurrent analysis |
| Timeout | 300s | Full stock analysis takes 60–90s |
| Workers | 1 | InMemorySessionService is per-process |
