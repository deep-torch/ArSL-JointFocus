
import os
import argparse
import zipfile
from concurrent.futures import ThreadPoolExecutor

import py7zr
import gdown


def download_dataset(files_ids, output_dir, file_extension='zip'):
    outputs = []
    for idx, file_id in enumerate(files_ids):
        output_path = os.path.join(output_dir, f'temp_{idx}.{file_extension}')
        gdown.download(id=file_id, output=output_path, quiet=False)
        outputs.append(output_path)

    return outputs


def extract(file_path):
    extract_path = os.path.dirname(file_path)

    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif file_path.endswith('.7z'):
        with py7zr.SevenZipFile(file_path, mode='r') as z:
            z.extractall(extract_path)
    else:
        raise ValueError(f"File '{file_path}' has unsupported archive file.")

    os.remove(file_path)


def recursive_extract(root_dir):
    data_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.7z'):
                data_files.append(os.path.join(root, f))

    with ThreadPoolExecutor() as executor:
        executor.map(extract, data_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract dataset.')

    parser.add_argument('--files_ids', nargs='+', required=True, help='Google Drive files IDs')
    parser.add_argument('--extract_dir', type=str, help='Directory path to extract the files to')

    args = parser.parse_args()

    outputs = download_dataset(args.files_ids, args.extract_dir)

    with ThreadPoolExecutor() as executor:
        executor.map(extract, outputs)

    recursive_extract(args.extract_dir)

    # download labels
    gdown.download(id=args.labels_file_id, output=os.path.join(args.extract_dir, 'labels.xlsx'), quiet=False)

    print(f'Successfully downloaded dataset to {args.extract_dir}')
