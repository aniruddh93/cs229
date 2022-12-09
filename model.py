# Project main model file

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

import fc_nn

#train_pos_path = "/Users/aniruddhr/Documents/cs229/project/aclImdb/train/pos"
#train_neg_path = "/Users/aniruddhr/Documents/cs229/project/aclImdb/train/neg"
#test_pos_path = "/Users/aniruddhr/Documents/cs229/project/aclImdb/test/pos"
#test_neg_path = "/Users/aniruddhr/Documents/cs229/project/aclImdb/test/neg"

train_pos_path = "/home/aniruddh_ramrakhyani/my_proj/cs229/aclImdb/train/pos"
train_neg_path = "/home/aniruddh_ramrakhyani/my_proj/cs229/aclImdb/train/neg"
test_pos_path = "/home/aniruddh_ramrakhyani/my_proj/cs229/aclImdb/test/pos"
test_neg_path = "/home/aniruddh_ramrakhyani/my_proj/cs229/aclImdb/test/neg"



def GetWords(lines):
    """
    Converts list of sentences to list of words with all words converted to small case.
    Ignores one-letter words (i.e. doesn't include them.
    
    Args:
      data: list of sentences in a review.

    Returns:
      list of words in review (all coverted to small case)
    """

    words = []
    for l in lines:
        temp = l.split(" ")
        for t in temp:
            if len(t) > 1:
                words.append(t.lower())

    return words


def CreateDict():
    """Creates a vocab by parsing words in train dataset.

    Returns:
      vocab as a dict mapping word to unique idx   
    """
    train_pos_dir_list = os.listdir(train_pos_path)
    train_neg_dir_list = os.listdir(train_neg_path)

    skip_words = [
        "this", "is", "that", "the", "a", "of", "to", "it", "in", "i", "an", "and", "at", "!",
        "was", "on", "he", "his", "have", "you", "her", "their", "me", "we", "i'm", "am", "mr",
        "i'd", "you'll", "i'll"
    ]

    vocab_count = {}

    # parse positive reviews
    for f in train_pos_dir_list:
        file_path = os.path.join(train_pos_path, f)

        fr = open(file_path, "r")
        data = fr.readlines()

        words = GetWords(data)
        for w in words:
            if w not in skip_words:
                vocab_count[w] = vocab_count.get(w, 0) + 1

        fr.close()

    # parse negative reviews
    for f in train_neg_dir_list:
        file_path = os.path.join(train_neg_path, f)

        fr = open(file_path, "r")
        data = fr.readlines()

        words = GetWords(data)
        for w in words:
            if w not in skip_words:
                vocab_count[w] = vocab_count.get(w, 0) + 1

        fr.close()
    

    # prune words that occur in less than five reviews
    word_prune_list = []
    idx = 0
    for w in vocab_count:
        if vocab_count[w] < 5:
            word_prune_list.append(w)
        else:
            vocab_count[w] = idx
            idx += 1

    for pw in word_prune_list:
        del vocab_count[pw]

    return vocab_count


def CreateBagOfWordsFeatures(vocab, train=True):
    """Creates bag of words features for the reviews given a vocabulary.

    Args:
      vocab: List of words in vocabulary.

    Returns:
      bag_of_words: a numpy matrix with bag of words representation for each review as a row.
      Y: true labels
    """

    pos_dir_list = []
    neg_dir_list = []
    pos_path = ""
    neg_path = ""

    if train:
        pos_dir_list = os.listdir(train_pos_path)
        neg_dir_list = os.listdir(train_neg_path)
        pos_path = train_pos_path
        neg_path = train_neg_path
    else:
        pos_dir_list = os.listdir(test_pos_path)
        neg_dir_list = os.listdir(test_neg_path)
        pos_path = test_pos_path
        neg_path = test_neg_path

    num_pos_ex = len(pos_dir_list)
    num_neg_ex = len(neg_dir_list)
    num_train_ex = num_pos_ex + num_neg_ex

    Y = np.zeros((num_train_ex, 1), dtype=float)
    Y[0:num_pos_ex][:] = 1.0
    
    bag_of_words = np.zeros((num_train_ex, len(vocab)), dtype=int)
    ex_idx = 0

    for pos_file in pos_dir_list:
        file_path = os.path.join(pos_path, pos_file)

        fr = open(file_path, "r")
        data = fr.readlines()
        
        words = GetWords(data)
        for w in words:
            if w in vocab:
                idx = vocab[w]
                bag_of_words[ex_idx][idx] += 1

        ex_idx += 1
        fr.close()

    
    for neg_file in neg_dir_list:
        file_path = os.path.join(neg_path, neg_file)

        fr = open(file_path, "r")
        data = fr.readlines()
        
        words = GetWords(data)
        for w in words:
            if w in vocab:
                idx = vocab[w]
                bag_of_words[ex_idx][idx] += 1

        ex_idx += 1
        fr.close()    

    return (bag_of_words, Y)



def CreateInformationRetreivalFeatures(vocab, train=True):
    """ Creates (term_frequency * inverse_document_frequency) (tf-idf) feature for the dataset.

    Args:
      vocab: List of words in vocabulary.
    
    Returns:
      ir_features: a numpy matrix with information retrieval feature (tf-idf) representation for each review as a row.
      Y: true labels  
    """

    pos_dir_list = []
    neg_dir_list = []

    if train:
        pos_dir_list = os.listdir(train_pos_path)
        neg_dir_list = os.listdir(train_neg_path)
    else:
        pos_dir_list = os.listdir(test_pos_path)
        neg_dir_list = os.listdir(test_neg_path)

    num_examples = len(pos_dir_list) + len(neg_dir_list)

    # bag_of_words is the same as term frequency
    bag_of_words, Y = CreateBagOfWordsFeatures(vocab, train)
    vocab_len = bag_of_words.shape[1]

    zeros = np.zeros(bag_of_words.shape)
    ones = np.ones(bag_of_words.shape)

    # 1.0 for each word that occurs in the document
    doc_with_term = np.where(bag_of_words > 0, ones, zeros)

    # add 1 for numeric stabilization
    term_doc_freq = (np.sum(doc_with_term, axis=0) + 1.0) / (num_examples + 1)
    idf = np.log(term_doc_freq) * -1.0
    idf = idf.reshape(1, vocab_len)

    word_count_in_examples = np.sum(bag_of_words, axis=1).reshape(num_examples, 1)
    term_freq = bag_of_words / word_count_in_examples

    #print('bag_of_words shape: %s' % (bag_of_words.shape, ) )
    #print(bag_of_words[1, :15])
    #print('word_count_in_examples shape: %s', word_count_in_examples.shape)
    #print(word_count_in_examples[1, :15])
    #print('term freq shape: %s' % (term_freq.shape,) )
    #print(term_freq[1, :15])
    #print('idf shape: %s' % (idf.shape,) )
    #print(idf[0, :15])

    tf_idf = term_freq * idf

    #print('tf_idf shape: %s' % (tf_idf.shape,) )
    #print(tf_idf[1, :15])

    return tf_idf, Y



def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def PredictLogisticRegression(X, theta):
    return sigmoid(np.matmul(X, theta))


def TrainLogisticRegression(X, Y):
    """Trains a logistic regression model on data X

    Args:
      X: np array (batch_size, num_features). Should include bias term as the first column.
      Y: True labels

    Returns:
      theta : trained parameters
    """

    (batch_size, num_features) = X.shape
    lr = 1.0
    min_step = 1e-15
    max_iter = 1e3
    iter_num = 0
    
    #theta = np.zeros((num_features, 1), dtype=float)
    #theta = np.random.rand(num_features, 1)
    theta = np.random.exponential(scale=0.01, size=(num_features, 1))

    step_norm = 1.0
    training_loss = []

    
    while step_norm > min_step:
        prediction = PredictLogisticRegression(X, theta)  # (b, 1)
        step = np.transpose(np.matmul(np.transpose(Y - prediction), X)) / batch_size  # (f, 1)
        theta = theta + lr * step
        iter_num += 1
        step_norm = np.linalg.norm(step)

        # calculate loss every 5 iterations
        if iter_num % 5 == 0:
            loss_vec = (Y * np.log(prediction)) + ((1-Y) * np.log(1 - prediction))  # (b, 1)
            loss = np.sum(loss_vec) / batch_size
            training_loss.append(loss)
            print("loss %f, step_norm: %f, iter_num: %d" % (loss, step_norm, iter_num))

        if iter_num >= max_iter:
            print("breaking out: reached max iter count")
            break
    
    return (theta, training_loss)


def TrainLinearNeuralNet(X_train, Y_train, X_test, Y_test):
    """Trains a Linear NN model.

    Args:
      X: np array (batch_size, num_features). Should NOT include bias term as the first column.
      Y: True labels

    Returns:
      theta: trained parameters
    """

    num_layers = 2
    layer_dim = [2048, 1024]
    lr = 0.01
    batch_size = 1000
    num_epochs = 10
    

    model = fc_nn.FullyConnectedNN(num_layers, layer_dim)
    optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=lr)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    lh = fc_nn.LossHistory()
    
    model.compile(optimizer=optimizer, loss=bce, steps_per_execution=batch_size)
    model.fit(X_train, Y_train, batch_size, num_epochs, callbacks=[lh])

    train_prediction_raw = model.predict(X_train, batch_size)
    test_prediction_raw = model.predict(X_test, batch_size)
    
    GetMetrics(train_prediction_raw, Y_train, 'train')
    GetMetrics(test_prediction_raw, Y_test, 'test')

    return lh.losses
    


def AddBias(features):
    batch_size, num_features = features.shape
    biases = np.ones((batch_size, 1), dtype=float)
    new_features = np.concatenate((biases, features), axis=1)
    return new_features


def plot_loss(loss, save_path):
    """Plots loss with num of iterations on X-axis.

    Args:
      loss: vector of loss values.
    """

    cleaned_loss = [x for x in loss if x != float('NaN')]
    x = [i*5 for i in range(len(cleaned_loss))]

    plt.figure()
    plt.plot(x, cleaned_loss, linewidth=2)
    plt.xlabel("Num Iterations -->")
    plt.ylabel("loss")
    plt.savefig(save_path)

    
def LoadDataSet(vocab, feature_type = 'bag_of_words', train=True, add_bias=True):
    """Loads the test set.

    Returns:
     X: test features data
     Y: true labels
    """

    if(feature_type == 'bag_of_words'):
        (X, Y) = CreateBagOfWordsFeatures(vocab, train)
    elif(feature_type == 'tf_idf'):
        (X, Y) = CreateInformationRetreivalFeatures(vocab, train)
    else:
        assert False, "unknown feature type"

    if add_bias:
        X_new = AddBias(X)
        return (X_new, Y)
    else:
        return (X, Y)
        

def GetMetrics(predictions_raw, Y, prefix):
    """Returns metrics for the dataset.

    Args:
      theta: logistic regression parameters
      X: features
      Y: true labels

    Returns:
      precision
      accuracy
      recall
    """
    predictions = [1.0 if p > 0.5 else 0.0 for p in predictions_raw]
    Y = np.squeeze(Y)
    y_list = Y.tolist()
    batch_size = len(predictions)

    epsilon = 1e-10
    num_correct = 0
    for i in range(len(predictions)):
        if abs(predictions[i] - y_list[i]) < epsilon:
            num_correct += 1

    accuracy = num_correct / batch_size

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(predictions)):
        if abs(predictions[i] - 1.0) < epsilon and abs(y_list[i] - 1.0) < epsilon:
            true_positive += 1
        elif abs(predictions[i] - 1.0) < epsilon and abs(y_list[i]) < epsilon:
            false_positive += 1
        elif abs(predictions[i]) < epsilon and abs(y_list[i] - 1.0) < epsilon:
            false_negative += 1

    precision = float(true_positive) / float(true_positive + false_positive)
    recall = float(true_positive) / float(true_positive + false_negative)

    print("%s accuracy: %f, precision: %f, recall: %f" % (prefix, accuracy, precision, recall))

    return (accuracy, precision, recall)
            
     
        

def main():

    vocab_path = "/home/aniruddh_ramrakhyani/my_proj/cs229/aclImdb/my_vocab.txt"
    #vocab_path = "/Users/aniruddhr/Documents/cs229/project/aclImdb/my_vocab.txt"

    
    #plot_save_path = "/Users/aniruddhr/Documents/cs229/project/aclImdb/loss.png"
    plot_save_path = "/home/aniruddh_ramrakhyani/my_proj/cs229/aclImdb/loss.png"

    create_vocab = False
    vocab = {}
    
    feature_type = 'tf_idf'         # tf_idf, bag_of_words
    model_type = 'linear_nn'     # logistic_reg, linear_nn

    if create_vocab:
        vocab = CreateDict()
        with open(vocab_path, "w") as fw:
            fw.write(str(vocab))
    else:
        with open(vocab_path, "r") as fr:
            data = fr.read().rstrip()
            vocab = eval(data)

    np.random.seed(7)

    if model_type == 'logistic_reg':
        (X_train, Y_train) = LoadDataSet(vocab, feature_type, train=True, add_bias=True)
        (theta, training_loss) = TrainLogisticRegression(X_train, Y_train)
        plot_loss(training_loss, plot_save_path)
        predictions_raw_train = PredictLogisticRegression(X_train, theta)
        (train_acc, train_precision, train_recall) = GetMetrics(predictions_raw_train, Y_train, 'train')

        (X_test, Y_test) = LoadDataSet(vocab, train=False, add_bias=True)
        predictions_raw_test = PredictLogisticRegression(X_test, theta)
        (test_acc, test_precision, test_recall) = GetMetrics(predictions_raw_test, Y_test, 'test')

    elif model_type == 'linear_nn':
        (X_train, Y_train) = LoadDataSet(vocab, feature_type, train=True, add_bias=False)
        (X_test, Y_test) = LoadDataSet(vocab, train=False, add_bias=False)
        training_loss = TrainLinearNeuralNet(X_train, Y_train, X_test, Y_test)
        plot_loss(training_loss, plot_save_path)

    
    

if __name__ == "__main__":
    main()
