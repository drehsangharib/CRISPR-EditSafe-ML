#!/usr/bin/env python
from __future__ import annotations

import argparse
from crispr_editsafe.report import generate_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_html_report(args.metrics, args.predictions, args.output)
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
