# API Documentation

**Base URL:** `http://localhost:5000`
**Authentication:** None (Open API)

## Endpoints

### 1. Search API
**Endpoint:** `GET /api/v1/search`
Perform a search query against the corpus.

- **Parameters:**
  - `q` (string): The search query.
  - `method` (string): Search method (e.g., `bm25`, `semantic`, `hybrid`, `bm25_ltr`).
  - `top_k` (int): Number of top results to return.
  - `page` (int): Pagination offset.
  - `limit` (int): Results per page.
- **Response Schema:** JSON object containing `results`, `total`, `latency_ms`.
- **Curl Example:**
  ```bash
  curl "http://localhost:5000/api/v1/search?q=python&method=bm25&top_k=10"
  ```

### 2. Suggest API
**Endpoint:** `GET /api/v1/suggest`
Get query suggestions and spell corrections.

- **Parameters:**
  - `q` (string): Incomplete or misspelled query.
- **Response Schema:** JSON object containing an array of `suggestions`.
- **Curl Example:**
  ```bash
  curl "http://localhost:5000/api/v1/suggest?q=pyton"
  ```

### 3. Explain API
**Endpoint:** `GET /api/v1/explain`
Explain the scoring of a specific document for a query.

- **Parameters:**
  - `q` (string): The search query.
  - `doc` (string): Document ID.
- **Response Schema:** JSON object detailing term weights and final score.
- **Curl Example:**
  ```bash
  curl "http://localhost:5000/api/v1/explain?q=python&doc=doc_123"
  ```

### 4. List Experiments
**Endpoint:** `GET /api/v1/experiments`
List all active A/B experiments.

- **Response Schema:** JSON array of experiment objects.
- **Curl Example:**
  ```bash
  curl "http://localhost:5000/api/v1/experiments"
  ```

### 5. Get Experiment Details
**Endpoint:** `GET /api/v1/experiments/<id>`
Get details and variant statistics for a specific experiment.

- **Response Schema:** JSON object with experiment data.
- **Curl Example:**
  ```bash
  curl "http://localhost:5000/api/v1/experiments/exp_1"
  ```

### 6. Analytics Click Logging
**Endpoint:** `POST /api/v1/analytics/click`
Log a user click on a search result.

- **Body Parameters:**
  - `query` (string)
  - `doc_id` (string)
  - `rank` (int)
  - `session_id` (string)
- **Response Schema:** `{ "status": "success" }`
- **Curl Example:**
  ```bash
  curl -X POST "http://localhost:5000/api/v1/analytics/click" -H "Content-Type: application/json" -d '{"query":"python","doc_id":"doc_123","rank":1,"session_id":"abc"}'
  ```

### 7. Analytics CTR
**Endpoint:** `GET /api/v1/analytics/ctr`
Retrieve aggregate Click-Through Rate metrics.

- **Response Schema:** JSON object with CTR statistics.
- **Curl Example:**
  ```bash
  curl "http://localhost:5000/api/v1/analytics/ctr"
  ```

### 8. Health Check
**Endpoint:** `GET /api/v1/health`
Liveness probe.

- **Response Schema:** `{ "status": "ok" }`
- **Curl Example:**
  ```bash
  curl "http://localhost:5000/api/v1/health"
  ```

### 9. Readiness Check
**Endpoint:** `GET /api/v1/ready`
Readiness probe verifying index and dependencies are loaded.

- **Response Schema:** `{ "status": "ready" }`
- **Curl Example:**
  ```bash
  curl "http://localhost:5000/api/v1/ready"
  ```

### 10. Metrics (Prometheus)
**Endpoint:** `GET /api/v1/metrics`
Retrieve application metrics in Prometheus format.

- **Response Schema:** Plain text Prometheus metrics.
- **Curl Example:**
  ```bash
  curl "http://localhost:5000/api/v1/metrics"
  ```

## Error Response Format
All API errors return a standard JSON structure:
```json
{
  "error": "Error type or code",
  "message": "Human-readable description"
}
```

## Rate Limiting
The API is rate-limited to **120 requests per minute per IP address**. Exceeding this limit will result in a `429 Too Many Requests` HTTP response.
