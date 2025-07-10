#!/usr/bin/env python3
"""Test script to count Australian businesses in the registry."""

import sys

sys.path.append("/Users/tod/Desktop/unified_vision_processor")

from vision_processor.compliance.australian_business_registry import (
    AustralianBusinessRegistry,
)


def main():
    registry = AustralianBusinessRegistry()
    registry.initialize()

    stats = registry.get_business_statistics()

    print(f"Total Australian businesses in registry: {stats['total_businesses']}")
    print("\nBusinesses by industry:")
    for industry, count in stats["businesses_by_industry"].items():
        print(f"  {industry}: {count}")

    print("\nBusinesses by type:")
    for biz_type, count in stats["businesses_by_type"].items():
        print(f"  {biz_type}: {count}")

    print("\nGoal: 100+ businesses")
    print(
        f"Achieved: {stats['total_businesses']}/100+ ✓"
        if stats["total_businesses"] >= 100
        else f"Need: {100 - stats['total_businesses']} more"
    )


if __name__ == "__main__":
    main()
