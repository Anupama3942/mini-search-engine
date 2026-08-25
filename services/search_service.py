"""
Mini Search Engine - Stage 16 & 17
Production Search Service & Query Understanding Orchestrator
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional

import config
from search import SearchEngine, generate_snippet
from services.metrics import metrics_registry
from services.retrieval import BM25Retriever, SemanticRetriever, HybridRetriever
from ranking.ltr import LTRRanker
from ranking import get_ranker
from query_understanding import QueryUnderstandingPipeline, QueryRepresentation

logger = logging.getLogger("search_engine")


class SearchService:
    """
    Central search service coordinating query understanding, multi-stage retrieval,
    ranking, score attribution, pagination, metrics recording, and fallback handling.
    """

    _instance = None

    def __init__(self, engine: Optional[SearchEngine] = None):
        self.engine = engine or SearchEngine()
        self.bm25_retriever = BM25Retriever(self.engine)
        self.semantic_retriever = SemanticRetriever()
        self.hybrid_retriever = HybridRetriever(self.engine)
        self.ltr_ranker = LTRRanker()
        
        # Initialize Query Understanding Pipeline
        doc_titles = [f.replace(".txt", "").capitalize() for f in self.engine.documents.keys()]
        self.qu_pipeline = QueryUnderstandingPipeline(
            vocabulary=set(self.engine.inverted_index.keys()),
            term_frequencies=self.engine.doc_freq,
            document_titles=doc_titles
        )

    @classmethod
    def get_instance(cls) -> "SearchService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def validate_request(self, query: str, top_k: int, page: int, limit: int) -> Optional[str]:
        """Validate search request parameters against production safety boundaries."""
        if not query or not query.strip():
            return "Query string must not be empty."
        if len(query) > config.MAX_QUERY_LENGTH:
            return f"Query length ({len(query)}) exceeds maximum allowed ({config.MAX_QUERY_LENGTH} characters)."
        if top_k < config.MIN_TOP_K or top_k > config.MAX_TOP_K:
            return f"Parameter 'top_k' must be between {config.MIN_TOP_K} and {config.MAX_TOP_K}."
        if page < 1:
            return "Parameter 'page' must be >= 1."
        if limit < 1 or limit > config.MAX_TOP_K:
            return f"Parameter 'limit' must be between 1 and {config.MAX_TOP_K}."
        return None

    def get_suggestions(self, prefix: str, limit: int = 5) -> List[str]:
        """Autocomplete suggestions for search box."""
        return self.qu_pipeline.suggester.suggest(prefix, limit=limit)

    def search(
        self,
        query: str,
        method: Optional[str] = None,
        top_k: int = config.DEFAULT_TOP_K,
        page: int = 1,
        limit: int = config.DEFAULT_PAGE_SIZE,
        alpha: Optional[float] = None,
        debug: bool = False,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute search request through query understanding, strategy routing, or two-stage pipelines.
        """
        req_id = request_id or uuid.uuid4().hex[:8]
        t_start = time.perf_counter()
        raw_method = (method or config.DEFAULT_RANKING_ALGORITHM).lower().strip()

        # 1. Parameter Validation
        error_msg = self.validate_request(query, top_k, page, limit)
        if error_msg:
            metrics_registry.record_request(raw_method, 0.0, success=False, error_type="validation_error")
            return {
                "request_id": req_id,
                "error": error_msg,
                "status_code": 400
            }

        # 2. Query Understanding Pipeline Analysis (Stage 17)
        query_repr = self.qu_pipeline.analyze(query)
        
        # Adaptive Query Routing (if method is 'adaptive' or 'auto')
        if raw_method in ("adaptive", "auto", "routed"):
            effective_method = query_repr.suggested_strategy
        else:
            effective_method = raw_method

        search_query_text = query_repr.effective_query or query

        # 3. Pipeline Execution
        try:
            if effective_method in ("bm25_ltr", "bm25->ltr"):
                results = self._execute_two_stage_bm25_ltr(search_query_text, top_k=top_k)
            elif effective_method in ("hybrid_ltr", "hybrid->ltr"):
                results = self._execute_two_stage_hybrid_ltr(search_query_text, alpha=alpha, top_k=top_k)
            else:
                # Standard ranking strategies (bm25, tfidf, frequency, ltr, semantic, hybrid)
                raw_results = self.engine.search(
                    query=search_query_text,
                    top_k=top_k,
                    ranking_algorithm=effective_method,
                    alpha=alpha,
                    log_analytics=True
                )
                if isinstance(raw_results, dict) and "error" in raw_results:
                    raise ValueError(raw_results["error"])
                results = [
                    {
                        "filename": r["filename"],
                        "title": r.get("title", r["filename"]),
                        "snippet": r.get("snippet", ""),
                        "score": round(float(r["score"]), 4),
                        "ranking_algorithm": r.get("ranking_algorithm", effective_method)
                    }
                    for r in raw_results
                ]

            # Apply field filter if present (e.g. title:python)
            if query_repr.field_filters and "title" in query_repr.field_filters:
                expected_title = query_repr.field_filters["title"].lower()
                results = [r for r in results if expected_title in r["title"].lower() or expected_title in r["filename"].lower()]

            total_duration = round(time.perf_counter() - t_start, 6)
            metrics_registry.record_request(effective_method, total_duration, success=True)

            # 4. Pagination Slicing
            total_count = len(results)
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            paged_items = results[start_idx:end_idx]
            total_pages = max(1, (total_count + limit - 1) // limit)

            if config.LOG_QUERIES:
                logger.info(f"[{req_id}] search query='{query}' effective='{search_query_text}' method={effective_method} results={total_count} latency={total_duration*1000:.2f}ms")

            response_data = {
                "request_id": req_id,
                "query": query,
                "effective_query": search_query_text,
                "method": effective_method,
                "total_results": total_count,
                "page": page,
                "limit": limit,
                "total_pages": total_pages,
                "search_duration_seconds": total_duration,
                "results": paged_items
            }

            # Optional Debug View
            if debug:
                response_data["query_understanding"] = query_repr.to_dict()

            return response_data

        except Exception as e:
            total_duration = round(time.perf_counter() - t_start, 6)
            logger.error(f"[{req_id}] search error: {e}", exc_info=False)
            metrics_registry.record_request(effective_method, total_duration, success=False, error_type=type(e).__name__)
            
            # Fallback to BM25 if advanced pipeline failed
            if effective_method != "bm25":
                logger.warning(f"[{req_id}] Executing safe fallback to BM25 ranking.")
                return self.search(query=query, method="bm25", top_k=top_k, page=page, limit=limit, request_id=req_id)

            return {
                "request_id": req_id,
                "error": "Internal search processing error.",
                "status_code": 500
            }

    def _execute_two_stage_bm25_ltr(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Two-stage pipeline: BM25 candidate retrieval (pool=50) -> LTR Reranking (top_k)."""
        candidates = self.bm25_retriever.retrieve(query, top_k=config.CANDIDATE_POOL_SIZE)
        if not candidates:
            return []
        
        query_terms = query.lower().split()
        reranked = self.ltr_ranker.rank(query_terms, candidates, self.engine, top_k=top_k)
        
        return [
            {
                "filename": r["filename"],
                "title": r["filename"].replace(".txt", "").capitalize(),
                "snippet": generate_snippet(self.engine.documents.get(r["filename"], ""), query_terms),
                "score": round(float(r["score"]), 4),
                "ranking_algorithm": "bm25->ltr"
            }
            for r in reranked
        ]

    def _execute_two_stage_hybrid_ltr(self, query: str, alpha: Optional[float], top_k: int) -> List[Dict[str, Any]]:
        """Two-stage pipeline: Hybrid candidate retrieval (pool=50) -> LTR Reranking (top_k)."""
        candidates = self.hybrid_retriever.retrieve(query, top_k=config.CANDIDATE_POOL_SIZE)
        if not candidates:
            return []
        
        query_terms = query.lower().split()
        reranked = self.ltr_ranker.rank(query_terms, candidates, self.engine, top_k=top_k)
        
        return [
            {
                "filename": r["filename"],
                "title": r["filename"].replace(".txt", "").capitalize(),
                "snippet": generate_snippet(self.engine.documents.get(r["filename"], ""), query_terms),
                "score": round(float(r["score"]), 4),
                "ranking_algorithm": "hybrid->ltr"
            }
            for r in reranked
        ]
