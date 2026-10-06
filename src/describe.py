"""This program take a dataset as a parameter and display informations for all numerical features."""

import argparse

from stats import get_stats
from utils import load_csv


def main() -> int:
    """Display informations for all numerical features if valid file is given."""
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path of the file")
    args = parser.parse_args()
    data_frame = load_csv(args.file)
    if data_frame is None:
        return 0
    stats = get_stats(data_frame)
    print(stats)
    return 0


if __name__ == "__main__":
    main()
