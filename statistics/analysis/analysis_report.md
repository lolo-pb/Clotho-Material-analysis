# Segmentation statistical analysis

Analysed 31 masks in 3 magnification group(s).
Percentages are calculated per image; summary statistics therefore describe image-to-image variation.

## Results by magnification

| Magnification | Class | n | Mean (%) | SD | Median | Q1 | Q3 | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10X | fibers | 10 | 38.84 | 6.91 | 40.77 | 33.85 | 43.47 | 27.59 | 47.54 |
| 10X | resin | 10 | 46.81 | 7.12 | 46.64 | 40.44 | 53.18 | 38.17 | 57.25 |
| 10X | pores | 10 | 14.35 | 2.35 | 13.19 | 12.41 | 16.43 | 12.12 | 18.07 |
| 10X | undefined | 10 | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 |
| 20X | fibers | 11 | 54.51 | 5.51 | 55.71 | 50.31 | 57.53 | 46.59 | 63.75 |
| 20X | resin | 11 | 29.49 | 1.98 | 29.72 | 28.74 | 30.32 | 26.05 | 32.57 |
| 20X | pores | 11 | 16.00 | 5.56 | 14.84 | 12.59 | 19.69 | 7.83 | 27.33 |
| 20X | undefined | 11 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 | 0.03 |
| 50X | fibers | 10 | 58.83 | 9.07 | 57.36 | 55.00 | 64.26 | 40.91 | 72.12 |
| 50X | resin | 10 | 30.01 | 4.97 | 28.68 | 28.18 | 30.28 | 23.78 | 38.81 |
| 50X | pores | 10 | 10.78 | 7.74 | 8.83 | 5.84 | 13.40 | 2.76 | 29.79 |
| 50X | undefined | 10 | 0.39 | 0.40 | 0.24 | 0.10 | 0.73 | 0.00 | 1.05 |

## Files

- `sample_statistics.csv`: source data with inferred magnification.
- `group_summary.csv`: descriptive results used by the charts.
- `class_boxplots.png`: distribution, median, quartiles, and range for each class.
- `class_means.png`: mean ± one standard deviation for each class.
