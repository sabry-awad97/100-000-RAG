"""
Load testing script for RAG system.

This script performs load testing to measure system performance
under various query loads.
"""

import time
import asyncio
import statistics
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadTester:
    """
    Load testing for RAG system.

    Measures:
    - Query throughput
    - Latency distribution
    - Error rate
    - Cache performance
    """

    def __init__(self, pipeline):
        """
        Initialize load tester.

        Args:
            pipeline: RAG pipeline instance
        """
        self.pipeline = pipeline
        self.results: List[Dict] = []

    def run_test(
        self, queries: List[str], num_iterations: int = 10, concurrent: bool = False
    ) -> Dict:
        """
        Run load test.

        Args:
            queries: List of test queries
            num_iterations: Number of iterations per query
            concurrent: Whether to run queries concurrently

        Returns:
            Test results dictionary
        """
        logger.info(
            f"Starting load test with {len(queries)} queries, {num_iterations} iterations"
        )

        start_time = time.time()

        if concurrent:
            results = self._run_concurrent(queries, num_iterations)
        else:
            results = self._run_sequential(queries, num_iterations)

        total_time = time.time() - start_time

        return self._calculate_metrics(results, total_time)

    def _run_sequential(self, queries: List[str], num_iterations: int) -> List[Dict]:
        """Run queries sequentially."""
        results = []

        for iteration in range(num_iterations):
            for query in queries:
                result = self._execute_query(query)
                results.append(result)

        return results

    def _run_concurrent(self, queries: List[str], num_iterations: int) -> List[Dict]:
        """Run queries concurrently."""
        # Simplified synchronous version
        # In production, use asyncio or threading
        return self._run_sequential(queries, num_iterations)

    def _execute_query(self, query: str) -> Dict:
        """Execute single query and measure performance."""
        start_time = time.time()
        error = None

        try:
            result = self.pipeline.query(query)
            success = True
        except Exception as e:
            logger.error(f"Query failed: {e}")
            result = None
            success = False
            error = str(e)

        latency = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "query": query,
            "latency_ms": latency,
            "success": success,
            "error": error,
            "cache_hit": result.get("cache_hit", False) if result else False,
        }

    def _calculate_metrics(self, results: List[Dict], total_time: float) -> Dict:
        """Calculate performance metrics."""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        cache_hits = [r for r in results if r.get("cache_hit", False)]

        latencies = [r["latency_ms"] for r in successful]

        metrics = {
            "total_queries": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "error_rate": len(failed) / len(results) if results else 0,
            "cache_hit_rate": len(cache_hits) / len(results) if results else 0,
            "total_time_s": total_time,
            "throughput_qps": len(results) / total_time if total_time > 0 else 0,
        }

        if latencies:
            metrics.update(
                {
                    "latency_avg_ms": statistics.mean(latencies),
                    "latency_median_ms": statistics.median(latencies),
                    "latency_p95_ms": self._percentile(latencies, 95),
                    "latency_p99_ms": self._percentile(latencies, 99),
                    "latency_min_ms": min(latencies),
                    "latency_max_ms": max(latencies),
                }
            )

        return metrics

    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def print_results(self, metrics: Dict):
        """Print test results."""
        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)
        print(f"Total Queries:      {metrics['total_queries']}")
        print(f"Successful:         {metrics['successful']}")
        print(f"Failed:             {metrics['failed']}")
        print(f"Error Rate:         {metrics['error_rate']:.2%}")
        print(f"Cache Hit Rate:     {metrics['cache_hit_rate']:.2%}")
        print(f"Total Time:         {metrics['total_time_s']:.2f}s")
        print(f"Throughput:         {metrics['throughput_qps']:.2f} queries/sec")

        if "latency_avg_ms" in metrics:
            print("\nLatency Statistics:")
            print(f"  Average:          {metrics['latency_avg_ms']:.2f}ms")
            print(f"  Median:           {metrics['latency_median_ms']:.2f}ms")
            print(f"  P95:              {metrics['latency_p95_ms']:.2f}ms")
            print(f"  P99:              {metrics['latency_p99_ms']:.2f}ms")
            print(f"  Min:              {metrics['latency_min_ms']:.2f}ms")
            print(f"  Max:              {metrics['latency_max_ms']:.2f}ms")

        print("=" * 60)


def main():
    """Main entry point."""
    # Example test queries
    test_queries = [
        "What is the capital of France?",
        "Explain quantum computing",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
        "What is machine learning?",
    ]

    # Note: This requires a configured pipeline
    # from rag_system.pipelines import RAGPipeline
    # pipeline = RAGPipeline.create_from_settings()

    # tester = LoadTester(pipeline)
    # metrics = tester.run_test(test_queries, num_iterations=10)
    # tester.print_results(metrics)

    logger.info("Load testing script - configure pipeline to run")


if __name__ == "__main__":
    main()
