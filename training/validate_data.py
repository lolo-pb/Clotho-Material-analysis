import argparse
from pathlib import Path

from common.validation import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate painted segmentation masks.")
    parser.add_argument("--images", type=Path, default=Path("training/images"))
    parser.add_argument("--masks", type=Path, default=Path("training/masks"))
    parser.add_argument(
        "--clean-output", type=Path, default=Path("training/validated_masks")
    )
    parser.add_argument(
        "--with-overlays",
        action="store_true",
        help="Allow mixed exact masks and corrected overlays in the masks folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_dataset(
        args.images,
        args.masks,
        args.clean_output,
        with_overlays=args.with_overlays,
    )

    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")

    if not result.ok:
        print(f"Validation failed with {len(result.errors)} error(s).")
        return 1

    print(f"Validated {len(result.pairs)} image/mask pair(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
