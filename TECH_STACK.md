# Technology Stack

This document details the technologies chosen for the Mini Search Engine project and the rationale behind each choice.

- **Python 3.12+**
  - *Why:* Provides an extensive standard library and modern language features, keeping the project educational and minimizing external dependencies.
- **Flask 3.0.3**
  - *Why:* A lightweight, micro web framework perfectly suited for exposing REST APIs without unnecessary overhead.
- **SQLite**
  - *Why:* Zero-configuration, serverless database ideal for storing analytics and telemetry without the burden of setting up a standalone DB server.
- **Pure Python IR**
  - *Why:* Inverted index, BM25, and TF-IDF are built from scratch. This enforces educational goals by eliminating "black box" solutions like Elasticsearch or Lucene.
- **Pure Python Embeddings**
  - *Why:* Implements deterministic 64-dimensional embeddings without needing PyTorch, TensorFlow, or a GPU, making it extremely portable.
- **Pure Python LTR**
  - *Why:* Logistic regression for Learning-to-Rank is implemented natively in Python, eliminating the need for `scikit-learn` or heavy ML dependencies.
- **Jinja2**
  - *Why:* Fast, secure server-rendered templates for the minimal HTML UI and analytics dashboard.
- **tracemalloc**
  - *Why:* Built-in memory profiling to monitor memory utilization safely in production.
- **hashlib**
  - *Why:* Used to implement deterministic SHA-256 hashing for consistent, stateless A/B testing bucket assignment.
- **unittest**
  - *Why:* The standard Python testing library, sufficient for rigorous unit and integration testing without needing pytest.
- **Docker**
  - *Why:* Standardizes the application environment, making it easy to deploy, scale, and isolate.
- **systemd**
  - *Why:* Robust process management, auto-restarting, and security sandboxing on Linux deployments.
- **Prometheus format**
  - *Why:* Exposing metrics in the widely adopted Prometheus standard ensures the system is observable and ready for modern monitoring stacks.
