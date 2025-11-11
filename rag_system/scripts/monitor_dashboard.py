"""
Optional metrics dashboard for monitoring RAG system performance.

This script provides a simple dashboard for visualizing system metrics.
"""

import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsDashboard:
    """
    Simple metrics dashboard for RAG system.

    Displays real-time metrics including:
    - Query throughput
    - Average latency
    - Cache hit rate
    - User satisfaction
    """

    def __init__(self, monitor):
        """
        Initialize dashboard.

        Args:
            monitor: RAGMonitor instance
        """
        self.monitor = monitor
        self.running = False

    def start(self, refresh_interval: int = 5):
        """
        Start dashboard display.

        Args:
            refresh_interval: Refresh interval in seconds
        """
        self.running = True

        logger.info("Starting metrics dashboard...")
        logger.info(f"Refresh interval: {refresh_interval}s")

        try:
            while self.running:
                self._display_metrics()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
            self.running = False

    def stop(self):
        """Stop dashboard."""
        self.running = False

    def _display_metrics(self):
        """Display current metrics."""
        metrics = self.monitor.get_metrics_summary(time_window=3600)

        print("\n" + "=" * 50)
        print("RAG SYSTEM METRICS (Last Hour)")
        print("=" * 50)
        print(f"Total Queries:     {metrics['total_queries']}")
        print(f"Avg Latency:       {metrics['avg_latency_ms']:.2f}ms")
        print(f"Cache Hit Rate:    {metrics['cache_hit_rate']:.1%}")
        print(f"Avg User Rating:   {metrics['avg_user_rating']:.2f}/5.0")
        print("=" * 50)


def main():
    """Main entry point for dashboard."""
    from rag_system.core.monitoring import RAGMonitor

    monitor = RAGMonitor()
    dashboard = MetricsDashboard(monitor)

    try:
        dashboard.start(refresh_interval=5)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")


if __name__ == "__main__":
    main()
