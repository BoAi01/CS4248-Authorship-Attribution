# General

# Visualization

# Feature extraction approach
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Classification

# BERT classifier
# Installing a custom version of Simple Transformers
# !git clone https://github.com/NVIDIA/apex
# !pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
#!git init
# !pip install --upgrade tqdm
# !git remote add origin https://github.com/ThilinaRajapakse/simpletransformers.git
# !git pull origin master
# !pip install -r requirements-dev.txt
# !pip install transformers
# !pip install tensorboardX

# !pip install simpletransformers

# # Parallelize apply on Pandas
# !pip install pandarallel
from pandarallel import pandarallel
pandarallel.initialize()

# Evaluation

if __name__ == '__main__':
    import argparse
    from train import run_iterations

    datasets = ['imdb62', 'enron', 'imdb', 'blog']
    parser = argparse.ArgumentParser(description=f'Training models for datasets {datasets}')
    parser.add_argument('--dataset', type=str, help='the dataset used for training')
    parser.add_argument('--mode', type=str, default='bert_ensemble', help='the dataset used for training')
    args = parser.parse_args()
    print(args)
    if args.mode == 'bert_ensemble':
        run_iterations_bert(source=args.dataset)
    else:
        run_iterations(source=args.dataset)
