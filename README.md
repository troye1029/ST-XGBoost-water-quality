# ST-XGBoost-water-quality
Spatio-temporal prediction and future scenario analysis of heterogeneous water quality indicators
in the Huaihe River Basin using adaptive machine learning.

The Google Earth Engine scripts in `GEE_code/` build the predictor table from open archives —
Sentinel-2 Level-2A for NDCI, NDTI, FUI and NDVI, ERA5-Land for temperature and precipitation,
Dynamic World for cropland and built-up fractions, VIIRS for night lights, NASADEM for terrain and
JRC for water occurrence — and export one 100 m CSV per year for 2019-2025. `basemap.txt` and
`landuse.txt` produce the true-colour basemap and the annual land-cover maps used in the figures.
Raster data are not stored here; the scripts regenerate everything from the original archives.

The Python code trains and evaluates the model. `train_RFE_Optuna_XGBoost.py` is the entry point:
it loads the table, fits a default XGBoost baseline, applies recursive feature elimination, tunes
the hyperparameters with Optuna under a chronological split, refits the final model and writes the
metrics, figures and model files. `features.py`, `modeling.py` and `plotting.py` contain the
feature engineering, the tuning routines and the figures. 2025 is held out for testing and all
earlier years are used for training.

Merge the exported CSVs into one table with `longitude`, `latitude`, `year`, `NDCI`, `NDTI`, `FUI`
and the predictor columns, then run:

```bash
pip install -r requirements.txt
python code/train_RFE_Optuna_XGBoost.py --data path/to/table.csv --target ALL
```
