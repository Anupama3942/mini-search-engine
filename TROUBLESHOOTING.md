# Troubleshooting & Operational Guide

This document outlines common operational issues, diagnostics, and recovery procedures.

---

### 1. Issue: Vector Index Missing or Corrupted
- **Symptoms**: `GET /ready` returns HTTP 503 or logs indicate `VectorStore Warning: Failed to load vector index`.
- **Cause**: Vector index JSON was deleted or unbuilt.
- **Resolution**:
  ```bash
  python build_index.py
  ```
  The engine will automatically fall back to BM25 in the interim without failing user queries.

---

### 2. Issue: LTR Model Missing or Feature Version Mismatch
- **Symptoms**: Logs show `Feature version mismatch: model=X, system=1.0`.
- **Cause**: Feature definitions were updated without retraining.
- **Resolution**:
  ```bash
  python train_ltr.py
  ```
  The system automatically uses BM25 fallback until retraining completes.

---

### 3. Issue: Rate Limit Exceeded (HTTP 429)
- **Symptoms**: Client receives `{"error": "Too Many Requests: Rate limit exceeded."}`.
- **Cause**: Client IP exceeded `RATE_LIMIT_REQUESTS` per minute.
- **Resolution**: Increase `RATE_LIMIT_REQUESTS` in `.env` or implement client-side backoff:
  ```env
  RATE_LIMIT_REQUESTS=300
  ```

---

### 4. Issue: Port Already in Use (Errno 48 / 98)
- **Symptoms**: `OSError: [Errno 98] Address already in use: 5000`.
- **Resolution**: Change port in `.env` or terminate the conflicting process:
  ```bash
  export PORT=8080
  python app.py
  ```

---

### 5. Disaster Recovery Procedure
If all indexes become corrupted or unusable:
1. Ensure raw `.txt` files exist in `documents/`.
2. Run full rebuild:
   ```bash
   python build_index.py
   python train_ltr.py
   ```
3. Restart application:
   ```bash
   # Systemd:
   sudo systemctl restart search-engine
   # Docker:
   docker restart search-engine
   ```
4. Verify readiness:
   ```bash
   curl -i http://localhost:5000/ready
   ```
