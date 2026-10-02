"""Bundled example datasets for training demos."""

from importlib.resources import files

REVIEWS_CSV = files(__package__) / "reviews.csv"


def export(dest: str = "reviews.csv") -> str:
    """Write the bundled reviews.csv to *dest* and return the path."""
    from pathlib import Path

    src = REVIEWS_CSV.read_text()
    p = Path(dest)
    p.write_text(src)
    return str(p)
