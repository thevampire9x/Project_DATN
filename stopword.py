from sklearn.feature_extraction.text import TfidfVectorizer

class StopWord:
    def get_stopwords(self,documents, threshold=3):
        """
        :param documents: list of documents
        :param threshold:
        :return: list of words has idf <= threshold
        """
        tfidf = TfidfVectorizer(min_df=10)
        tfidf_matrix = tfidf.fit_transform(documents)
        features = tfidf.get_feature_names()
        stopwords = []
        for index, feature in enumerate(features):
            if tfidf.idf_[index] <= threshold:
                stopwords.append(feature)
        return stopwords

    def get_stopwords_fromfile(self):
        with open('vietnamese_stopwords.txt', 'r', encoding='utf8') as f:
            stopwords = f.readline()
            stop_set = set(m.strip() for m in stopwords)
            return list(frozenset(stop_set))

    def remove_stopwords(self,line, stopword):
        words = []
        for word in line.strip().split():
            if word not in stopword:
                words.append(word)
        return ' '.join(words)
