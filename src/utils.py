import pandas as pd
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

def create_logs(data, features, replace = False):
    exprs = []
    for var in features:
        ln = (pl.col(var).abs()+1).log()
        col_name = var if replace else f"Log_{var}"
        exprs.append(ln.alias(col_name))
    return data


def main():
    pass

if __name__ == "__main__":
    main()