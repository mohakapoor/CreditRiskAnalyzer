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
        
    df.to_parquet(output_dir)    
    print(f"converted {csv_path} to parquet")

def main():
    pass

if __name__ == "__main__":
    main()