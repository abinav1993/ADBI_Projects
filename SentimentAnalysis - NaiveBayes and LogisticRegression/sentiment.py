import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

def removeStopwords(data, stopwords):
    temp = []
    for wordsList in data:
        temp.append(filter(lambda x: x not in stopwords,wordsList))
    return temp    

def buildVectors(data, features):
    result = []
    for wordList in data:
        temp = [0]*len(features.keys())
        result.append(temp)
    
    for i,wordList in enumerate(data):
        for word in wordList:
            if word in features:
                result[i][features[word]] = 1
                
    #print result[0]
    return result

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    train_pos = removeStopwords(train_pos, stopwords)
    train_neg = removeStopwords(train_neg, stopwords)
    test_pos = removeStopwords(test_pos, stopwords)
    test_neg = removeStopwords(test_neg, stopwords)

    min_pos = 0.01 * len(train_pos)
    min_neg = 0.01 * len(train_neg)
    print min_pos
    wordcount = {}

    for wordList in train_pos:
        for word in set(wordList):
            if word in wordcount:
                wordcount[word][0] = wordcount[word][0] + 1
            else:
                wordcount[word] = [1,0]

    for wordList in train_neg:
        for word in set(wordList):
            if word in wordcount:
                wordcount[word][1] = wordcount[word][1] + 1
            else:
                wordcount[word] = [0,1]
    print len(wordcount)

    features = {}
    i = 0
    for word,counts in wordcount.items():
        if counts[0] >= min_pos or counts[1] >= min_neg:
            if counts[0] >= 2*counts[1] or counts[1] >= 2*counts[0]:
                features[word] = i
                i += 1
            elif counts[0] == 0 or counts[1] == 0:
                features[word] = i
                i += 1

    print len(features)
    
    train_pos_vec = buildVectors(train_pos, features)
    train_neg_vec = buildVectors(train_neg, features)
    test_pos_vec = buildVectors(test_pos, features)
    test_neg_vec = buildVectors(test_neg, features)

    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    labeled_train_pos = []
    labeled_train_neg = []
    labeled_test_pos = []
    labeled_test_neg = []

    for i in range(len(train_pos)):
        labeled_train_pos.append(LabeledSentence(words = train_pos[i], tags = ['TRAIN_POS_'+str(i)] ))

    for i in range(len(train_neg)):
        labeled_train_neg.append(LabeledSentence(words = train_neg[i], tags = ['TRAIN_NEG_'+str(i)] ))

    for i in range(len(test_pos)):
        labeled_test_pos.append(LabeledSentence(words = test_pos[i], tags = ['TEST_POS_'+str(i)] ))

    for i in range(len(test_neg)):
        labeled_test_neg.append(LabeledSentence(words = test_neg[i], tags = ['TEST_NEG_'+str(i)] ))

    print len(labeled_train_pos)
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

   
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for i in range(len(labeled_train_pos)):
        train_pos_vec.append(model.docvecs['TRAIN_POS_'+str(i)])

    for i in range(len(labeled_train_pos)):
        train_neg_vec.append(model.docvecs['TRAIN_NEG_'+str(i)])

    for i in range(len(labeled_train_pos)):
        test_pos_vec.append(model.docvecs['TEST_POS_'+str(i)])

    for i in range(len(labeled_train_pos)):
        test_neg_vec.append(model.docvecs['TEST_NEG_'+str(i)])
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X,Y)

    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1.0,binarize=None)
    nb_model.fit(X,Y)
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X,Y)
    
    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit(X,Y)

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    Y = ["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec)

    test = test_pos_vec + test_neg_vec
    result = model.predict(test)
    
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(Y)):
        if result[i] == Y[i] and Y[i] == 'pos':
             tp += 1
        if result[i] == 'pos' and Y[i] == 'neg':
             fp += 1
        if result[i] == Y[i] and Y[i] == 'neg':
             tn += 1
        if result[i] == 'neg' and Y[i] == 'pos':
             fn += 1
    
    accuracy = (tp + tn) * 1.0 / (tp + tn + fp + fn)
    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
