import re
from ast import literal_eval
from vncorenlp import VnCoreNLP
import os
import datetime
from random import shuffle

# rdrsegmenter = VnCoreNLP("/data2/nghiatl/REC/news-recommendation/src/VNCORENLP/VnCoreNLP-1.2.jar", annotators="wseg,pos,ner,parse", max_heap_size='-Xmx2g')
rdrsegmenter = VnCoreNLP(address="http://10.5.1.230", port=2311)
print('Vncorenlp loaded')

uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ."
punc_re = '[^\w'+uniChars+']'

def shuffle_list(row):
    list_clicked_0 =  row['clicked'].split()
    list_can = [row['candidate_news']] + row['click_0_itemIDs'] 
    l =list(zip(list_can, list_clicked_0))
    shuffle(l)
    a, b = zip(*l)
    row['candidate_news'] = list(a)
    row['clicked'] = ' '.join(list(b))
    
    return row

def take_k_clicked_0_news(click_0_itemIDs, K):
    return click_0_itemIDs[:K]

def update_phase(file_path, new_contents):
    # Open the file for writing (this will overwrite the existing contents)
    with open(file_path, "w") as file:
        file.write(new_contents)
    print("File overwritten successfully.")
    
def filter_not_exist(row, news_df):
    a_set = set(row['clicked_news'])
    exist_list = news_df['newsId'].tolist()
    if (a_set.issubset(exist_list)): 
#     and (set(row['click_0_itemIDs']).issubset(exist_list)) and (row['candidate_news'] in exist_list):
        return True
    return False

def latest_checkpoint(directory, serving=False):
    if not os.path.exists(directory):
        return None
    
    lisr_dir = os.listdir(directory)
    timestamp_folders = [folder for folder in lisr_dir if folder.count('-') >= 2]
    timestamp_folders.sort(reverse=True)
    if serving == True:
        with open(os.path.join(directory, 'training.txt'), 'r') as f:
            phase = f.read()

        if phase == 'True':
            timestamp_folders = timestamp_folders[1:]
        
    print(timestamp_folders)
    if len(timestamp_folders) == 0:
        return None
    # Find the latest folder based on timestamps
    latest_folder = max(timestamp_folders, key=lambda folder: datetime.datetime.strptime(folder, "%Y-%m-%d_%H-%M-%S"))
    
    if serving == True:
        for t_fold in timestamp_folders:
            t_dir = os.path.join(directory, t_fold)
            all_ckpt = [x for x in os.listdir(t_dir) if '.pth' in x]
            print(all_ckpt)
            if len(all_ckpt) > 0:
                latest_folder = t_fold
                break
            
    print("Latest folder:", os.path.join(directory, latest_folder))
    directory = os.path.join(directory, latest_folder)
    
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory) if '.pth' in x
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])

def segment_text(sen):
    ##Lower
    try:
        sen = sen.lower()
    except:
        print(sen)
    ##Segment
    seg = rdrsegmenter.tokenize(sen)
    if len(seg) == 0:
        sen = ' '
    else: 
        sen = ' '.join(seg[0])
    return sen

def striphtml(data, DEBUG=False, url = ''):
   
    try:
        assert isinstance(data, str), 'Not String'
    except:
        print(data)
    p = re.compile(r'<.*?>')
    p = p.sub(' ', data)
    p = re.sub('\s+',' ', p)
    p = re.sub('&nbsp;','', p)
    p = p.replace("&amp;", " ")
    p = re.sub(r'https?:\/\/.*[\r\n]*', '', p, flags=re.MULTILINE)

    return p.strip()

def preprocess_text(sen):
    if not isinstance(sen, str):
        sen = ''
    ##Unicode reform
    assert isinstance(sen, str), 'Not String'
    # sen = convert_unicode(sen)
    sen = striphtml(sen)
    ##Remove punctuation
    sen = re.sub(punc_re,' ', sen)
    ##Remove multiple space
    sen = re.sub('\s+',' ', sen)
    ##Lower
    sen = sen.strip()
    sen = re.sub(r'\.', ' .', sen)
    ##Segment
    # seg = rdrsegmenter.tokenize(sen)
    # if len(seg) == 0:
    #   sen = ' '
    # else: 
    #   sen = ' '.join(seg[0])
    return sen

def convert_string_to_list(string):
    return literal_eval(string)