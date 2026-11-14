# Geometric Proxy Coverage in European Land Cover Validation

This archive contains analysis scripts and derived data for the manuscript *Geometric Proxy Coverage Reveals Validation Sensitivity For WorldCover Across Europe*, by Junjie Zhang and Siyu Wang. It documents the corrected frame and both fixed samples used in the revision. The results describe geometric proxy coverage and scoring conventions for the analyzed observations; they are not estimates of European map accuracy.

## Contents

- `revision/results/corrected/`: filtering, corrected-frame summaries and the corrected 800-point sample.
- `revision/results/point_tables/`: the corrected full-frame point table, the original 800-point independent table, an original-sample country table and class-bridge candidate sets.
- `revision/results/new_sample/`: corrected 800-point sample, 1,599 location conditions and 799 paired comparisons.
- `revision/results/gps/` and `revision/results/radial/`: position and proxy-shape diagnostics for the historical 800-point sample.
- `data_processed/`: historical fixed-sample input and comparison summaries. The historical sample and corrected sample are distinct, and the historical sample was selected from the earlier frame.
- `revision/code/` and `scripts/`: analysis and checking scripts. `revision/inputs/` contains a compact cache extract required by some diagnostics. `requirements.txt` and `revision/requirements.txt` list the top-level dependencies used in the recorded environment.

The original source CSV and WorldCover rasters are not included. Files with point IDs and coordinate fields are linkable to public LUCAS records and should not be represented as anonymized. Several result JSON files retain recorded provenance; the local path fields in two files were replaced with the public input filename before release, without changing results.

## Original data and attribution

- European Commission, Joint Research Centre, LUCAS Copernicus 2022: https://doi.org/10.2905/JRC.XVTS8YG. The input was `l2022_survey_cop.csv`, available in the JRC data directory: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/LUCAS/LUCAS_2022_Copernicus/output/. The local input used for the analysis had SHA-256 `b5d5414fd8058bb1067496080883138a819700c21c9300c82f4a31cfe1528d6f`. See the directory's `copyright.txt` for the original licence and attribution notice.
- Zanaga et al., ESA WorldCover 10 m 2021 v200: https://doi.org/10.5281/zenodo.7254221. WorldCover is distributed under CC BY 4.0: https://esa-worldcover.org/en/data-access. Contains modified Copernicus Sentinel data (2021) processed by the ESA WorldCover consortium.

The derived tables contain selected observations, extracted map classes and calculated geometric summaries; they are modifications of, not replacements for, the original datasets. The authors' archive is made available under CC BY 4.0, with attribution to the original data providers as stated above. See https://creativecommons.org/licenses/by/4.0/.

## Offline verification

From the archive root, the following Python 3 commands inspect the supplied tables without accessing remote rasters:

```text
python scripts/independent_check.py
python revision/code/verify_revision.py
python revision/code/verify_new_sample.py
```

These tests verify selected sample sizes, joins, arithmetic and stored summary values. They do not independently reconstruct WorldCover pixels or all geometries. They were run with Python 3.12.10 in the existing project environment. A clean installation on another system has not been tested. Scripts to reconstruct every final manuscript figure are not all included; this archive does not claim complete automated figure regeneration.

To rerun the corrected full-frame geometry, obtain the source CSV independently and use a new output directory within `runs`:

```text
python revision/code/r3_corrected_full_geometry.py --raw PATH_TO_LUCAS_CSV --write-point-records --output-dir runs/my_corrected
python revision/code/stable_threshold_counts.py --points runs/my_corrected/r3_corrected_full_points.csv --output runs/my_corrected/stable_thresholds.json
```

`revision/code/new_sample/fetch_cog.py --help` describes the remote extraction inputs. Remote access to WorldCover COG tiles is required, along with the LUCAS source and geospatial dependencies. The recorded complete rerun used a compact cache for some conditions. It was not a completely cache-independent run. The radial geometry is an approximation, not an independently validated reproduction of an official radial polygon. The recorded environment was Windows with Python 3.12.10, pandas 3.0.5, NumPy 2.5.1, Shapely 2.1.2, pyproj 3.7.2 and rasterio 1.5.1; the requirements are top-level pins rather than a complete lock.

For the full-frame thresholds reported in the manuscript, round each fraction with pandas `round(10)` before applying the thresholds. Unrounded comparisons can differ at numerical boundaries. The corrected point table has 136,605 observations; the two samples each have 800 observations, of which 134 point IDs overlap. Observation-position comparisons and polygon-shape comparisons are sensitivity analyses, not measurements of GPS error or validation of true reference support.
