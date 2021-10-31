import gdown
import os

datasets = {
    'enron.csv': 'https://drive.google.com/uc?id=16Tzx6yxhsLG4gRuviI68cPIM2y3lVtjv&export=download',
    'full_enron.csv': 'https://drive.google.com/uc?id=1y0aJLI9JeWu3Vv7lnIZbPTqZjYCIG5GY&export=download',
    'full_imdb.csv': 'https://drive.google.com/uc?id=1bsC1DSYcg7FlmpvfrXJbIIfnBxqqKeWZ&export=download',
    'full_imdb_feat.csv': 'https://drive.google.com/uc?id=1Mq6Dt7m-imcP4a21a90H8IXrtaVUgUAN&export=download',
    'imdb62.txt': 'https://drive.google.com/uc?id=1t6P6I16i4LIHx1sldIqwMYqlBALI1JbN&export=download',
    'full_imdb62.csv': 'https://drive.google.com/uc?id=1neOw0rwXzEL4g-EaqaqJ1-jIGCFhDfZn&export=download',
    'blogtext.csv': 'https://drive.google.com/uc?id=1iZyaXG_M6vUjH1-uLSBnVAGCJyLodhgB&export=download',
    'full_blog.csv': 'https://drive.google.com/uc?id=1V8R3ZzH_hb97EOsT5BXECcEkgGU9NBkf&export=download'
}

dataset_dir = 'datasets'
os.makedirs(dataset_dir, exist_ok=True)

for name, link in datasets.items():
    if name in os.listdir(dataset_dir):
        continue
    gdown.download(link, os.path.join(dataset_dir, name), quiet=False)
