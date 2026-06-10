import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

def list_files(path,filetype):
    folder = Path(path)
    files,names = [],[]
    for file in folder.glob(f"*{filetype}"):
        files.append(file.name)
        names.append(file.stem)
    return dict(zip(files,names))

def csv_to_prqt(csv_path, output_dir):
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        print(f"UTF-8 decode failed for {csv_path}, trying 'latin1' encoding...")
        df = pd.read_csv(csv_path, encoding='latin1')

    df = data_downcasting(df)
    df.to_parquet(output_dir)    
    print(f"converted {csv_path} to parquet")

def data_downcasting(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            # Downcasting float64 to float32
            if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

            # Downcasting int64 to int32
            elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    return df

def count_missings(data):
    total_rows = len(data)
    null_data = []
    for col in data.columns:
        null_count = data[col].null_count()
        if null_count >0 :
            null_data.append(
                {
                    "Feature": col,
                    "Total": null_count,
                    "Percent": (null_count/ total_rows)*100,
                }
            )
    return pl.DataFrame(null_data).sort("Total",descending=True)
    
def convert_days(data, features, t = 12, rounding = True, replace = False):
    exprs = []
    for var in features:
        calc = -pl.col(var)/t
        if rounding:
            calc = calc.round(0)
        final_expr = pl.when(calc<0).then(None).otherwise(calc)
        col_name = var if replace else f"CONVERTED_{var}"
        exprs.append(final_expr.alias(col_name))
    return data.with_columns(exprs)

def create_logarithms(data, features, replace = False):
    exprs = []
    for var in features:
        ln = (pl.col(var).abs()+1).log()
        col_name = var if replace else f"Log_{var}"
        exprs.append(ln.alias(col_name))
    return data.with_columns(exprs)

def create_null_flags(data,features=None):
    if features is None:
        features = data.select(pl.all().exclude(pl.col("SK_ID_CURR"))).columns
    exprs = []
    for var in features:
        is_null = pl.when(pl.col(var).is_null()).then(1).otherwise(0).alias(f"ISNULL_{var}")
        exprs.append(is_null)
    return data.with_columns(exprs)

def treat_factors(data,method="label"):
    if method == "label":
        factor_cols = data.select(pl.col(pl.Categorical)).columns
        return data.with_columns(
            pl.col(factor_cols).cast(pl.Categorical)
        )
    elif method == "dummy":
        factor_cols = data.select(pl.col(pl.Categorical)).columns
        return data.with_columns(
            pl.col(factor_cols).to_dummies()
        )
    return data

def compute_accept_reject_ratio(data,lags = [1,3,5]):
    pass
    

def main():
    pass

if __name__ == "__main__":
    main()