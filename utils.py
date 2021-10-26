# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import itertools
from pandas import DataFrame

# Visualization
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
import plotly.express as px

# Feature extraction approach
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Classification
import xgboost as xgb
import lightgbm as lgbm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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
from simpletransformers.classification import ClassificationModel

import torch

# # Parallelize apply on Pandas
# !pip install pandarallel
from pandarallel import pandarallel
pandarallel.initialize()

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import os
from prepare_dataset import dataset_dir


def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
    plt.show()

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 11;
    figsize=[9,9];
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)


def run_iterations(source="enron", recompute=False):
    # Load data and remove emails containing the sender's name

    print("Loading and processing dataframe")

    ## Pre-processing functions
    def is_name_in_email(name, email):
        """
        Removing emails from Enron where name is in email
        """

        if str(name).lower() in str(email).lower():
            return 1
        else:
            return 0

    def fil_sent(sent):
        """
        Filter stopwords
        """

        filtered_sentence = ' '.join([w for w in sent.split() if not w in stop_words])
        return filtered_sentence

    def process(sent):
        """
        Apply stemming
        """

        sent = str(sent)
        return fil_sent(' '.join([ps.stem(str(x).lower()) for x in word_tokenize(sent)]))

    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    def extract_style(text):
        """
        Extracting stylometric features of a text
        """

        text = str(text)
        len_text = len(text)
        len_words = len(text.split())
        avg_len = np.mean([len(t) for t in text.split()])
        num_short_w = len([t for t in text.split() if len(t) < 3])
        per_digit = sum(t.isdigit() for t in text) / len(text)
        per_cap = sum(1 for t in text if t.isupper()) / len(text)
        f_a = sum(1 for t in text if t.lower() == "a") / len(text)
        f_b = sum(1 for t in text if t.lower() == "b") / len(text)
        f_c = sum(1 for t in text if t.lower() == "c") / len(text)
        f_d = sum(1 for t in text if t.lower() == "d") / len(text)
        f_e = sum(1 for t in text if t.lower() == "e") / len(text)
        f_f = sum(1 for t in text if t.lower() == "f") / len(text)
        f_g = sum(1 for t in text if t.lower() == "g") / len(text)
        f_h = sum(1 for t in text if t.lower() == "h") / len(text)
        f_i = sum(1 for t in text if t.lower() == "i") / len(text)
        f_j = sum(1 for t in text if t.lower() == "j") / len(text)
        f_k = sum(1 for t in text if t.lower() == "k") / len(text)
        f_l = sum(1 for t in text if t.lower() == "l") / len(text)
        f_m = sum(1 for t in text if t.lower() == "m") / len(text)
        f_n = sum(1 for t in text if t.lower() == "n") / len(text)
        f_o = sum(1 for t in text if t.lower() == "o") / len(text)
        f_p = sum(1 for t in text if t.lower() == "p") / len(text)
        f_q = sum(1 for t in text if t.lower() == "q") / len(text)
        f_r = sum(1 for t in text if t.lower() == "r") / len(text)
        f_s = sum(1 for t in text if t.lower() == "s") / len(text)
        f_t = sum(1 for t in text if t.lower() == "t") / len(text)
        f_u = sum(1 for t in text if t.lower() == "u") / len(text)
        f_v = sum(1 for t in text if t.lower() == "v") / len(text)
        f_w = sum(1 for t in text if t.lower() == "w") / len(text)
        f_x = sum(1 for t in text if t.lower() == "x") / len(text)
        f_y = sum(1 for t in text if t.lower() == "y") / len(text)
        f_z = sum(1 for t in text if t.lower() == "z") / len(text)
        f_1 = sum(1 for t in text if t.lower() == "1") / len(text)
        f_2 = sum(1 for t in text if t.lower() == "2") / len(text)
        f_3 = sum(1 for t in text if t.lower() == "3") / len(text)
        f_4 = sum(1 for t in text if t.lower() == "4") / len(text)
        f_5 = sum(1 for t in text if t.lower() == "5") / len(text)
        f_6 = sum(1 for t in text if t.lower() == "6") / len(text)
        f_7 = sum(1 for t in text if t.lower() == "7") / len(text)
        f_8 = sum(1 for t in text if t.lower() == "8") / len(text)
        f_9 = sum(1 for t in text if t.lower() == "9") / len(text)
        f_0 = sum(1 for t in text if t.lower() == "0") / len(text)
        f_e_0 = sum(1 for t in text if t.lower() == "!") / len(text)
        f_e_1 = sum(1 for t in text if t.lower() == "-") / len(text)
        f_e_2 = sum(1 for t in text if t.lower() == ":") / len(text)
        f_e_3 = sum(1 for t in text if t.lower() == "?") / len(text)
        f_e_4 = sum(1 for t in text if t.lower() == ".") / len(text)
        f_e_5 = sum(1 for t in text if t.lower() == ",") / len(text)
        f_e_6 = sum(1 for t in text if t.lower() == ";") / len(text)
        f_e_7 = sum(1 for t in text if t.lower() == "'") / len(text)
        f_e_8 = sum(1 for t in text if t.lower() == "/") / len(text)
        f_e_9 = sum(1 for t in text if t.lower() == "(") / len(text)
        f_e_10 = sum(1 for t in text if t.lower() == ")") / len(text)
        f_e_11 = sum(1 for t in text if t.lower() == "&") / len(text)
        richness = len(list(set(text.split()))) / len(text.split())

        return pd.Series(
            [avg_len, len_text, len_words, num_short_w, per_digit, per_cap, f_a, f_b, f_c, f_d, f_e, f_f, f_g, f_h, f_i,
             f_j, f_k, f_l, f_m, f_n, f_o, f_p, f_q, f_r, f_s, f_t, f_u, f_v, f_w, f_x, f_y, f_z, f_0, f_1, f_2, f_3,
             f_4, f_5, f_6, f_7, f_8, f_9, f_e_0, f_e_1, f_e_2, f_e_3, f_e_4, f_e_5, f_e_6, f_e_7, f_e_8, f_e_9, f_e_10,
             f_e_11, richness])

    list_senders = [5, 10, 25, 50, 75, 100]

    if source == "enron":

        if recompute:
            df = pd.read_csv(os.path.join(dataset_path, 'enron.csv'))
            df['name'] = df['From'].apply(lambda x: x.split("'")[1].split(".")[0])
            df['name_in_mail'] = df.apply(lambda x: is_name_in_email(x['name'], x['content']), axis=1)
            df = df[df['name_in_mail'] == 0]
            df = df[df['content'].apply(lambda x: '-----' not in str(x))]
            df = df[df['content'].apply(lambda x: "@" not in str(x))]
            df = df[df['content'].apply(lambda x: "From: " not in str(x))]
            df = df[df['content'].apply(lambda x: len(str(x).split()) > 10)]
            df['content_tfidf'] = df['content'].parallel_apply(lambda x: process(x))
            df[["avg_len", "len_text", "len_words", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d",
                "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s",
                "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7",
                "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9",
                "f_e_10", "f_e_11", "richness"]] = df['content'].parallel_apply(lambda x: extract_style(x))
            df.to_csv(os.path.join(dataset_path, "full_enron.csv"))
        else:
            df = pd.read_csv(os.path.join(dataset_path, 'full_enron.csv'))
            df['name'] = df['From'].apply(lambda x: x.split("'")[1].split(".")[0])
            df['name_in_mail'] = df.apply(lambda x: is_name_in_email(x['name'], x['content']), axis=1)
            df = df[df['name_in_mail'] == 0]
            df = df[df['content'].apply(lambda x: '-----' not in str(x))]
            df = df[df['content'].apply(lambda x: "@" not in str(x))]
            df = df[df['content'].apply(lambda x: "From: " not in str(x))]
            df = df[df['content'].apply(lambda x: len(str(x).split()) > 10)]
            df.to_csv(os.path.join(dataset_path, 'full_enron2.csv'))
            print(df.shape)

    elif source == "imdb":

        if recompute:
            df = pd.read_csv(os.path.join(dataset_dir, 'full_imdb.csv'), index_col=0)
            df['content_tfidf'] = df['content'].parallel_apply(lambda x: process(x))
            df[["avg_len", "len_text", "len_words", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d",
                "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s",
                "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7",
                "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9",
                "f_e_10", "f_e_11", "richness"]] = df['content'].parallel_apply(lambda x: extract_style(x))
            df.to_csv("full_imdb_feat.csv")
        else:
            df = pd.read_csv(os.path.join(dataset_dir, 'full_imdb_feat.csv'), index_col=0)

    elif source == "imdb62":

        list_senders = [62]

        if recompute:

            df = pd.read_csv(os.path.join(dataset_dir, "imdb62.txt"), sep="\t", header=None)
            df.columns = ["reviewid", "From", "Itemid", "grade", "title", "content"]
            df['content_tfidf'] = df['content'].parallel_apply(lambda x: process(x))
            df[["avg_len", "len_text", "len_words", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d",
                "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s",
                "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7",
                "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9",
                "f_e_10", "f_e_11", "richness"]] = df['content'].parallel_apply(lambda x: extract_style(x))
            df.to_csv(os.path.join(dataset_dir, "full_imdb62.txt"))

        else:

            df = pd.read_csv(os.path.join(dataset_dir, "full_imdb62.csv"), index_col=0)

    elif source == "blog":

        if recompute:
            df = pd.read_csv(os.path.join(dataset_dir, 'blogtext.csv'))
            df.columns = ["From", "Gender", "Age", "Topic", "Sign", "Date", "content"]
            df = df[df['content'].apply(lambda x: len(x.split())) > 0]
            df['content_tfidf'] = df['content'].parallel_apply(lambda x: process(x))
            df[["avg_len", "len_text", "len_words", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d",
                "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s",
                "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7",
                "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9",
                "f_e_10", "f_e_11", "richness"]] = df['content'].parallel_apply(lambda x: extract_style(x))
            df.to_csv("full_blog.csv")
        else:
            df = pd.read_csv(os.path.join(dataset_dir, 'full_blog.csv'))
            # Keep longer texts
            # df = df[df['content'].apply(lambda x: len(str(x).split()) > 30)]

    list_scores = []
    list_f1 = []

    import nltk
    from nltk.util import ngrams
    from collections import Counter
    import heapq

    def return_best_bi_grams(text):
        bigrams = ngrams(text, 2)

        data = dict(Counter(bigrams))
        list_ngrams = heapq.nlargest(100, data.keys(), key=lambda k: data[k])
        return list_ngrams

    def return_best_tri_grams(text):
        trigrams = ngrams(text, 3)

        data = dict(Counter(trigrams))
        list_ngrams = heapq.nlargest(100, data.keys(), key=lambda k: data[k])
        return list_ngrams

    def find_freq_n_gram_in_txt(text, list_bigram, list_trigram):

        to_ret = []

        num_bigrams = len(Counter(zip(text, text[1:])))
        num_trigrams = len(Counter(zip(text, text[1:], text[2:])))

        for n_gram in list_bigram:
            to_ret.append(text.count(''.join(n_gram)) / num_bigrams)

        for n_gram in list_trigram:
            to_ret.append(text.count(''.join(n_gram)) / num_trigrams)

        return to_ret

    for limit in list_senders:

        print("Number of speakers : ", limit)

        # Select top N senders and build Train and Test

        list_spk = list(pd.DataFrame(df['From'].value_counts()[:limit]).reset_index()['index'])
        sub_df = df[df['From'].isin(list_spk)]
        sub_df = sub_df[
            ['From', 'content', 'content_tfidf', "avg_len", "len_text", "len_words", "num_short_w", "per_digit",
             "per_cap", "f_a", "f_b", "f_c", "f_d", "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m",
             "f_n", "f_o", "f_p", "f_q", "f_r", "f_s", "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1",
             "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4",
             "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9", "f_e_10", "f_e_11", "richness"]]
        sub_df = sub_df.dropna()

        text = " ".join(sub_df['content'].values)
        list_bigram = return_best_bi_grams(text)
        list_trigram = return_best_tri_grams(text)

        print("Number of texts : ", len(sub_df))

        dict_nlp_enron = {}
        k = 0

        for val in np.unique(sub_df.From):
            dict_nlp_enron[val] = k
            k += 1

        sub_df['Target'] = sub_df['From'].apply(lambda x: dict_nlp_enron[x])

        ind = train_test_split(sub_df[['content', 'Target']], test_size=0.2, stratify=sub_df['Target'])
        ind_train = list(ind[0].index)
        ind_test = list(ind[1].index)

        nlp_train = sub_df.loc[ind_train]
        nlp_test = sub_df.loc[ind_test]

        # TF-IDF + LR

        print("#####")
        # print("Training TF-IDF")
        #
        # vectorizer = TfidfVectorizer()  # ngram_range=(1,2), max_features=3000
        # X_train = vectorizer.fit_transform(nlp_train['content_tfidf'])
        # X_test = vectorizer.transform(nlp_test['content_tfidf'])
        #
        # clf = LogisticRegression(random_state=0).fit(X_train, nlp_train['Target'])
        # y_pred = clf.predict(X_test)
        # score_lr = accuracy_score(nlp_test['Target'], y_pred)
        # f1_lr = f1_score(nlp_test['Target'], y_pred, average="macro")
        #
        # if limit < 11:
        #     plot_confusion_matrix_from_data(nlp_test['Target'], y_pred)
        #
        # print("Training done, accuracy is : ", score_lr)
        # print("Training done, f1-score is : ", f1_lr)

        # Bert + Classification Layer

        print("#####")
        print("Training BERT")

        n_gpu = 1    # torch.cuda.device_count()
        model = ClassificationModel('bert', 'bert-base-cased', num_labels=limit,
                                    args={'reprocess_input_data': True, 'overwrite_output_dir': True,
                                          'num_train_epochs': 15, 'n_gpu': n_gpu},
                                    use_cuda=True)
        model.train_model(nlp_train[['content', 'Target']])
        print(f'\n BERT model uses {n_gpu} GPUs\n')

        predictions, raw_outputs = model.predict(list(nlp_test['content']))
        score_bert = accuracy_score(predictions, nlp_test['Target'])
        f1_bert = f1_score(predictions, nlp_test['Target'], average="macro")

        if limit < 11:
            plot_confusion_matrix_from_data(nlp_test['Target'], predictions)

        predictions, raw_out_train = model.predict(list(nlp_train['content']))

        print("Training done, accuracy is : ", score_bert)
        print("Training done, f1-score is : ", f1_bert)

        # Style-based classifier

        print("#####")
        print("Training style classifier")

        X_style_train = nlp_train[
            ["avg_len", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d", "f_e", "f_f", "f_g", "f_h",
             "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s", "f_t", "f_u", "f_v", "f_w",
             "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_e_0",
             "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9", "f_e_10", "f_e_11",
             "richness"]]
        X_style_test = nlp_test[
            ["avg_len", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d", "f_e", "f_f", "f_g", "f_h",
             "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s", "f_t", "f_u", "f_v", "f_w",
             "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_e_0",
             "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9", "f_e_10", "f_e_11",
             "richness"]]

        # clf = xgb.XGBClassifier().fit(X_style_train, nlp_train['Target'])
        clf = LogisticRegression(random_state=0).fit(X_style_train, nlp_train['Target'])
        y_pred = clf.predict(X_style_test)
        y_proba = clf.predict_proba(X_style_test)
        y_proba_train = clf.predict_proba(X_style_train)
        score_style = accuracy_score(nlp_test['Target'], y_pred)
        f1_style = f1_score(nlp_test['Target'], y_pred, average="macro")

        print("Training done, accuracy is : ", score_style)
        print("Training done, f1-score is : ", f1_style)

        # Model Combination

        print("#####")
        print("Model combination")

        feat_for_BERT_LR_train = np.concatenate([raw_out_train, y_proba_train], axis=1)
        feat_for_BERT_LR_test = np.concatenate([raw_outputs, y_proba], axis=1)

        clf = LogisticRegression(random_state=0).fit(feat_for_BERT_LR_train, nlp_train['Target'])
        # clf = xgb.XGBClassifier().fit(feat_for_BERT_LR_train, nlp_train['Target'])

        y_pred = clf.predict(feat_for_BERT_LR_test)
        score_comb = accuracy_score(nlp_test['Target'], y_pred)
        f1_comb = f1_score(nlp_test['Target'], y_pred, average="macro")

        print("Training done, accuracy is : ", score_comb)
        print("Training done, f1-score is : ", f1_comb)

        if limit < 11:
            plot_confusion_matrix_from_data(nlp_test['Target'], y_pred)

        # Character N-gram only

        print("#####")
        print("Character N-gram")

        feats_train = nlp_train['content'].apply(lambda x: find_freq_n_gram_in_txt(x, list_bigram, list_trigram)).values
        feats_test = nlp_test['content'].apply(lambda x: find_freq_n_gram_in_txt(x, list_bigram, list_trigram)).values

        feats_train = pd.DataFrame(feats_train)[0].apply(lambda x: pd.Series(x))
        feats_test = pd.DataFrame(feats_test)[0].apply(lambda x: pd.Series(x))

        # clf_char = xgb.XGBClassifier().fit(feats_train, nlp_train['Target'])
        clf_char = LogisticRegression(random_state=0).fit(feats_train, nlp_train['Target'])
        y_pred = clf_char.predict(feats_test)
        y_proba = clf_char.predict_proba(feats_test)
        y_proba_train = clf_char.predict_proba(feats_train)

        score_char = accuracy_score(nlp_test['Target'], y_pred)
        f1_char = f1_score(nlp_test['Target'], y_pred, average="macro")

        print("Training done, accuracy is : ", score_char)
        print("Training done, f1-score is : ", f1_char)

        # BERT + Style + Char N-gram

        print("#####")
        print("BERT + Style + Char N-gram")

        feat_for_BERT_full_train = np.concatenate([feat_for_BERT_LR_train, y_proba_train], axis=1)
        feat_for_BERT_full_test = np.concatenate([feat_for_BERT_LR_test, y_proba], axis=1)

        # clf = xgb.XGBClassifier().fit(feat_for_BERT_full_train, nlp_train['Target'])
        clf = LogisticRegression(random_state=0).fit(feat_for_BERT_full_train, nlp_train['Target'])

        y_pred = clf.predict(feat_for_BERT_full_test)
        score_comb_fin = accuracy_score(nlp_test['Target'], y_pred)
        f1_comb_fin = f1_score(nlp_test['Target'], y_pred, average="macro")
        print("Training done, accuracy is : ", score_comb_fin)
        print("Training done, f1-score is : ", f1_comb_fin)

        if limit < 11:
            plot_confusion_matrix_from_data(nlp_test['Target'], y_pred)

        # Store scores
        list_scores.append([limit, score_lr, score_bert, score_style, score_comb])
        list_f1.append([limit, f1_lr, f1_bert, f1_style, f1_comb])

    list_scores = np.array(list_scores)

    # Plot the output accuracy
    # plt.figure(figsize=(12, 8))
    # plt.plot(list_scores[:, 0], list_scores[:, 1], label="TF-IDF + LR")
    # plt.plot(list_scores[:, 0], list_scores[:, 2], label="Bert + Classification layer")
    # plt.plot(list_scores[:, 0], list_scores[:, 3], label="Stylometric")
    # plt.plot(list_scores[:, 0], list_scores[:, 4], label="Bert + Style")
    # plt.title("Classification Accuracy depending on the number of speakers")
    # plt.xlabel("Number of speakers")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()

    return list_scores, list_f1

