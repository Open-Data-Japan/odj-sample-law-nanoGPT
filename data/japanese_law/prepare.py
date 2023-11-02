import os
import gzip
import requests
import tiktoken
import numpy as np

# download japanese law text dataset from japan open data
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
data_urls = ['https://github.com/Open-Data-Japan/odj-data-e-gov-lawsearch/raw/main/data/processed/japanese_law_text.txt.gz.aa',
    'https://github.com/Open-Data-Japan/odj-data-e-gov-lawsearch/raw/main/data/processed/japanese_law_text.txt.gz.ab',
    'https://github.com/Open-Data-Japan/odj-data-e-gov-lawsearch/raw/main/data/processed/japanese_law_text.txt.gz.ac',
    'https://github.com/Open-Data-Japan/odj-data-e-gov-lawsearch/raw/main/data/processed/japanese_law_text.txt.gz.ad',
    'https://github.com/Open-Data-Japan/odj-data-e-gov-lawsearch/raw/main/data/processed/japanese_law_text.txt.gz.ae']
if not os.path.exists(input_file_path + '.gz'):
    for data_url in data_urls:
        local_filename = os.path.join(os.path.dirname(__file__),data_url.split('/')[-1])
        if not os.path.exists(local_filename):
            print(f'downloading {local_filename}')
            rq = requests.get(data_url, stream=True)
            with open(local_filename, 'wb') as f:
                for chunk in rq.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

# merge files
if not os.path.exists(input_file_path + '.gz'):
    print('merging')
    with open(input_file_path + '.gz', "wb") as f:
        for data_url in data_urls:
            cf = data_url.split('/')[-1]
            print(f'merging {cf}')
            local_filename = os.path.join(os.path.dirname(__file__),cf)
            with open(local_filename, 'rb') as g:
                f.write(g.read())

print('reading text file')
with gzip.open(input_file_path + '.gz', 'rt', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
