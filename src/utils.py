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

def main():
    pass

if __name__ == "__main__":
    main()