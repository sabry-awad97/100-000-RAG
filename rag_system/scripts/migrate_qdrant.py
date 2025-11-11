"""
Qdrant migration script for collection management.

This script helps with creating, updating, and migrating Qdrant collections.
"""

import argparse
import logging
from rag_system.core.indexing import DocumentIndexer, IndexConfig
from rag_system.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_collection(args):
    """Create new Qdrant collection."""
    logger.info(f"Creating collection: {args.collection_name}")

    config = IndexConfig(
        collection_name=args.collection_name,
        vector_size=args.vector_size,
        on_disk=args.on_disk,
    )

    indexer = DocumentIndexer(qdrant_url=settings.qdrant.url, config=config)
    indexer.create_collection(config)

    logger.info(f"Collection '{args.collection_name}' created successfully")


def delete_collection(args):
    """Delete Qdrant collection."""
    logger.warning(f"Deleting collection: {args.collection_name}")

    if not args.force:
        confirm = input(
            f"Are you sure you want to delete '{args.collection_name}'? (yes/no): "
        )
        if confirm.lower() != "yes":
            logger.info("Deletion cancelled")
            return

    indexer = DocumentIndexer(qdrant_url=settings.qdrant.url)
    indexer.delete_collection(args.collection_name)

    logger.info(f"Collection '{args.collection_name}' deleted")


def info_collection(args):
    """Display collection information."""
    logger.info(f"Getting info for collection: {args.collection_name}")

    indexer = DocumentIndexer(qdrant_url=settings.qdrant.url)
    info = indexer.get_collection_info(args.collection_name)

    print("\nCollection Information:")
    print("=" * 50)
    print(f"Name: {args.collection_name}")
    print(f"Info: {info}")
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Qdrant collection management")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create collection
    create_parser = subparsers.add_parser("create", help="Create new collection")
    create_parser.add_argument("collection_name", help="Collection name")
    create_parser.add_argument(
        "--vector-size", type=int, default=1536, help="Vector size"
    )
    create_parser.add_argument("--on-disk", action="store_true", help="Store on disk")

    # Delete collection
    delete_parser = subparsers.add_parser("delete", help="Delete collection")
    delete_parser.add_argument("collection_name", help="Collection name")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    # Collection info
    info_parser = subparsers.add_parser("info", help="Get collection info")
    info_parser.add_argument("collection_name", help="Collection name")

    args = parser.parse_args()

    if args.command == "create":
        create_collection(args)
    elif args.command == "delete":
        delete_collection(args)
    elif args.command == "info":
        info_collection(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
