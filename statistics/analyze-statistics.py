"""Create grouped descriptive reports and charts from segmentation statistics."""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


CLASS_COLUMNS = ("fibers", "resin", "pores", "undefined")
CLASS_COLORS_BGR = {
    "fibers": (0, 0, 220),
    "resin": (0, 180, 0),
    "pores": (220, 0, 0),
    "undefined": (70, 70, 70),
}
GROUP_PATTERN = re.compile(r"(?<!\d)(\d+)\s*[xX](?!\w)")


def read_statistics(input_path: Path) -> list[dict[str, object]]:
    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"{input_path} has no header row")
        missing_columns = set(CLASS_COLUMNS).difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_path} is missing required columns: {missing}")

        rows = []
        for row_number, row in enumerate(reader, start=2):
            try:
                values = {column: float(row[column]) for column in CLASS_COLUMNS}
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid percentage in row {row_number}") from error
            rows.append({**row, **values})
    if not rows:
        raise ValueError(f"{input_path} has no data rows")
    return rows


def infer_magnification(image_path: str) -> str:
    match = GROUP_PATTERN.search(Path(image_path).name)
    return f"{match.group(1)}X" if match else "Unspecified"


def percentile(values: np.ndarray, value: float) -> float:
    return float(np.percentile(values, value))


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": percentile(array, 25),
        "q3": percentile(array, 75),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def grouped_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = sorted({str(row["Magnification"]) for row in rows})
    summary_rows = []
    for group in groups:
        group_rows = [row for row in rows if row["Magnification"] == group]
        for class_name in CLASS_COLUMNS:
            summary_rows.append(
                {
                    "Magnification": group,
                    "Class": class_name,
                    **summarize([float(row[class_name]) for row in group_rows]),
                }
            )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_centered_text(
    image: np.ndarray, text: str, center_x: int, y: int, scale: float = 0.55
) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(
        image,
        text,
        (center_x - size[0] // 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )


def chart_canvas(title: str, width: int = 1400, height: int = 900) -> tuple[np.ndarray, int, int, int, int]:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, title, (70, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    left, top, right, bottom = 130, 105, width - 60, height - 130
    cv2.rectangle(image, (left, top), (right, bottom), (100, 100, 100), 1)
    return image, left, top, right, bottom


def draw_y_axis(image: np.ndarray, left: int, top: int, bottom: int) -> None:
    for percentage in range(0, 101, 20):
        y = int(bottom - (bottom - top) * percentage / 100)
        cv2.line(image, (left, y), (image.shape[1] - 60, y), (225, 225, 225), 1)
        cv2.putText(image, f"{percentage}%", (75, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)


def write_boxplot(rows: list[dict[str, object]], output_path: Path) -> None:
    image, left, top, right, bottom = chart_canvas("Segmentation percentages by magnification")
    draw_y_axis(image, left, top, bottom)
    groups = sorted({str(row["Magnification"]) for row in rows})
    group_width = (right - left) / len(groups)
    for group_index, group in enumerate(groups):
        group_rows = [row for row in rows if row["Magnification"] == group]
        center = int(left + group_width * (group_index + 0.5))
        for class_index, class_name in enumerate(CLASS_COLUMNS):
            values = np.asarray([float(row[class_name]) for row in group_rows])
            offset = int((class_index - 1.5) * 42)
            x = center + offset
            minimum, q1, median, q3, maximum = np.percentile(values, (0, 25, 50, 75, 100))
            y = lambda value: int(bottom - (bottom - top) * value / 100)
            color = CLASS_COLORS_BGR[class_name]
            cv2.line(image, (x, y(minimum)), (x, y(maximum)), color, 2)
            cv2.line(image, (x - 10, y(minimum)), (x + 10, y(minimum)), color, 2)
            cv2.line(image, (x - 10, y(maximum)), (x + 10, y(maximum)), color, 2)
            cv2.rectangle(image, (x - 14, y(q3)), (x + 14, y(q1)), color, -1)
            cv2.line(image, (x - 14, y(median)), (x + 14, y(median)), (255, 255, 255), 2)
        draw_centered_text(image, group, center, bottom + 32)
    for index, class_name in enumerate(CLASS_COLUMNS):
        x = left + index * 210
        cv2.rectangle(image, (x, 70), (x + 18, 88), CLASS_COLORS_BGR[class_name], -1)
        cv2.putText(image, class_name.title(), (x + 28, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def write_mean_chart(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    image, left, top, right, bottom = chart_canvas("Mean segmentation percentages by magnification")
    draw_y_axis(image, left, top, bottom)
    groups = sorted({str(row["Magnification"]) for row in summary_rows})
    group_width = (right - left) / len(groups)
    for group_index, group in enumerate(groups):
        values = {row["Class"]: row for row in summary_rows if row["Magnification"] == group}
        center = int(left + group_width * (group_index + 0.5))
        for class_index, class_name in enumerate(CLASS_COLUMNS):
            summary = values[class_name]
            bar_width = 28
            x = center + int((class_index - 1.5) * 38)
            y = int(bottom - (bottom - top) * float(summary["mean"]) / 100)
            cv2.rectangle(image, (x - bar_width // 2, y), (x + bar_width // 2, bottom), CLASS_COLORS_BGR[class_name], -1)
            error = (bottom - top) * float(summary["std"]) / 100
            cv2.line(image, (x, int(y - error)), (x, int(y + error)), (20, 20, 20), 1)
        draw_centered_text(image, group, center, bottom + 32)
    for index, class_name in enumerate(CLASS_COLUMNS):
        x = left + index * 210
        cv2.rectangle(image, (x, 70), (x + 18, 88), CLASS_COLORS_BGR[class_name], -1)
        cv2.putText(image, class_name.title(), (x + 28, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def write_report(path: Path, rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    groups = sorted({str(row["Magnification"]) for row in rows})
    lines = [
        "# Segmentation statistical analysis",
        "",
        f"Analysed {len(rows)} masks in {len(groups)} magnification group(s).",
        "Percentages are calculated per image; summary statistics therefore describe image-to-image variation.",
        "",
        "## Results by magnification",
        "",
        "| Magnification | Class | n | Mean (%) | SD | Median | Q1 | Q3 | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {Magnification} | {Class} | {count} | {mean:.2f} | {std:.2f} | {median:.2f} | {q1:.2f} | {q3:.2f} | {min:.2f} | {max:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `sample_statistics.csv`: source data with inferred magnification.",
            "- `group_summary.csv`: descriptive results used by the charts.",
            "- `class_boxplots.png`: distribution, median, quartiles, and range for each class.",
            "- `class_means.png`: mean ± one standard deviation for each class.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped descriptive statistics and charts from get-statistics.py CSV output."
    )
    parser.add_argument("input_csv", type=Path, help="CSV created by get-statistics.py")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("statistics/analysis"),
        help="Folder for CSV summaries, report, and PNG charts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_statistics(args.input_csv)
    for row in rows:
        row["Magnification"] = infer_magnification(str(row["Image Paths"]))
    summary_rows = grouped_summary(rows)
    output_dir = args.output_dir
    source_fields = list(rows[0].keys())
    write_csv(output_dir / "sample_statistics.csv", rows, source_fields)
    summary_fields = ["Magnification", "Class", "count", "mean", "std", "median", "q1", "q3", "min", "max"]
    write_csv(output_dir / "group_summary.csv", summary_rows, summary_fields)
    write_boxplot(rows, output_dir / "class_boxplots.png")
    write_mean_chart(summary_rows, output_dir / "class_means.png")
    write_report(output_dir / "analysis_report.md", rows, summary_rows)
    print(f"Wrote grouped analysis for {len(rows)} mask(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
