import os
from os import listdir,path
from os.path import isfile, join
import regex as re
import pickle
import time
import numpy as np
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from stopword import StopWord
from txt_preprocess import TextPreprocess

tp = TextPreprocess()
sw = StopWord()
def text_preprocess(document):
    # loại bỏ code html
    document = tp.remove_html(document)
    # chuẩn hóa unicode
    document = tp.convert_unicode(document)
    # chuẩn hóa cách gõ dấu tiếng Việt
    document = tp.chuan_hoa_dau_cau_tieng_viet(document)
    # tách từ
    document = word_tokenize(document, format="text")
    # đưa về lower
    document = document.lower()
    # xóa các ký tự không cần thiết
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', document)
    # loại bỏ chữ số
    document = re.sub(r'[0-9]+', '', document)
    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()
    return document

def read_files(inpath):
    documents = []
    inputfiles = [f for f in listdir(inpath) if isfile(join(inpath, f))]
    for file_name in inputfiles:
        f = open(inpath + file_name, 'r', encoding='utf8')
        documents.append(text_preprocess(f.read()))
        f.close()
    return documents

def file_processing(inpath, outpath):
    if not path.isdir(outpath)  : os.mkdir(outpath)
    else:
        for file in os.scandir(outpath):
            os.remove(file.path)
    documents = read_files(inpath)
    # stopwords = sw.get_stopwords(documents, 1)
    stopwords = sw.get_stopwords_fromfile()
    clean_docs = []
    for doc in documents:
        clean_docs.append(sw.remove_stopwords(doc, stopwords))
    i = 0
    output = 'out_text'
    for d in clean_docs:
        i = i + 1
        w = open(outpath + output + str(i) + '.txt', 'w', encoding='utf8')
        w.write(d)
        w.flush()
        w.close()
    print('Done word preprocessing!')

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

if __name__ == '__main__':
    inpath = 'Doc/'
    outpath ='output_tmpAll/'
    MODEL_PATH = 'model/'
    # file_processing(inpath, outpath)
    #

    test_percent = 0.2
    text = []
    label = []
    list_dir = os.listdir(outpath)
    documents = []
    for item in list_dir:
        if os.path.isdir(item):
            for f in read_files(outpath + item + '/'):
                documents.append(f)
                label.append(item)

    for line in documents:
        words = line.split()
        text.append(' '.join(words[0:]))

    X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent, random_state=42)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    #Naive Bayes
    start_time = time.time()
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                  max_df=0.8,
                                                  max_features=None)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())
                         ])
    text_clf = text_clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print('Done training Naive Bayes in', train_time, 'seconds.')

    # Save model
    if not path.isdir(MODEL_PATH): os.mkdir(MODEL_PATH)
    else:
        for file in os.scandir(MODEL_PATH):
            os.remove(file.path)

    pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'wb'))

    #Linear Regression
    start_time = time.time()
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                  max_df=0.8,
                                                  max_features=None)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression(solver='lbfgs',
                                                    multi_class='auto',
                                                    max_iter=10000))
                         ])
    text_clf = text_clf.fit(X_train, y_train)

    train_time = time.time() - start_time
    print('Done training Linear Classifier in', train_time, 'seconds.')

    # Save model
    pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "linear_classifier.pkl"), 'wb'))

    #SVM
    start_time = time.time()
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                  max_df=0.8,
                                                  max_features=None)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SVC(gamma='scale'))
                         ])
    text_clf = text_clf.fit(X_train, y_train)

    train_time = time.time() - start_time
    print('Done training SVM in', train_time, 'seconds.')

    # Save model
    pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "svm.pkl"), 'wb'))


    # Naive Bayes
    model = pickle.load(open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'rb'))
    y_pred = model.predict(X_test)
    print('Naive Bayes, Accuracy =', np.mean(y_pred == y_test))

    # Linear Classifier
    model = pickle.load(open(os.path.join(MODEL_PATH, "linear_classifier.pkl"), 'rb'))
    y_pred = model.predict(X_test)
    print('Linear Classifier, Accuracy =', np.mean(y_pred == y_test))

    # SVM
    model = pickle.load(open(os.path.join(MODEL_PATH, "svm.pkl"), 'rb'))
    y_pred = model.predict(X_test)
    print('SVM, Accuracy =', np.mean(y_pred == y_test))

    # Xem kết quả trên từng nhãn
    print('Bayes Naives')
    nb_model = pickle.load(open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'rb'))
    y_pred = nb_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=list(label_encoder.classes_)))
    print('Linear Classifier')
    nb_model = pickle.load(open(os.path.join(MODEL_PATH, "linear_classifier.pkl"), 'rb'))
    y_pred = nb_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=list(label_encoder.classes_)))
    print('SVM')
    nb_model = pickle.load(open(os.path.join(MODEL_PATH, "svm.pkl"), 'rb'))
    y_pred = nb_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=list(label_encoder.classes_)))