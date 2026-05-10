from utils import list_files,csv_to_prqt

def main():
    files = list_files('data/raw','csv')

    for i,j in files.items():
        input_path = f"data/raw/{i}"
        output_path = f"data/cleaned/{j.lower()}.parquet"
        csv_to_prqt(input_path,output_path)

if __name__ == "__main__":
    main()