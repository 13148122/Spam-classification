import collections

import numpy as np

import util


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    message = message.lower()
    words = message.split()
    # print("Words:",words)
    return words
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***

    all_word_list = []    # initialize empty list
    n = len(messages)
    # print("number of messages:", n)     #4457 total messages

    for i in np.arange(n):
        words = get_words(messages[i])
        all_word_list += words

    # print("all_word_list:",all_word_list)       #this worked
    # no=len(all_word_list)
    # print("Number of words in messages:",no)   #this worked- 70014 words

    dict_freq = {}
    j = 0
    for a_word in all_word_list:
        if (all_word_list.count(a_word) >= 5) and (a_word not in dict_freq):    # at least 5 word occurrences
            dict_freq[a_word] = j       # adding words to dictionary and mapping them as j
            j += 1

    # print(dict_freq)
    return dict_freq

    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    # print("word_dictionary:", word_dictionary)
    # print("Messages:  ",messages)

    n = len(messages)
    num = len(word_dictionary)
    word_freq = np.zeros((n, num))

    for j in np.arange(n):                # each value of j indicates each message
        words = get_words(messages[j])    # store the words of each message as a row
        for a_word in words:
            if a_word in word_dictionary.keys():    # checking if a_word is in the list of words in the given dictionary
                ind = word_dictionary[a_word]       # finding the index of a_word
                word_freq[j, ind] += 1

    # print("word_freq:", word_freq)
    print(len(word_freq))
    return word_freq

    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***

    model = {}
    n_mat = matrix.shape[1]         # 1757 elements- matches
    # print("matrix:", matrix)
    # print("n_mat:", n_mat)

    spam = matrix[labels == 1, :]   # storing the spam
    ham = matrix[labels == 0, :]    # storing the ham

    spam_lengths = spam.sum(axis=1)     # vertical sum axis=1
    ham_lengths = ham.sum(axis=1)       # vertical sum axis=1

    # the 1 and n_mat in numerator and denominator are for smoothing the naive bayes
    model['phi_spam'] = (spam.sum(axis=0) + 1) / (np.sum(spam_lengths) + n_mat)    # numerator sums pos occurrences
    model['phi_ham'] = (ham.sum(axis=0) + 1) / (np.sum(ham_lengths) + n_mat)       # numerator sums neg occurrences
    model['phi'] = spam.shape[0] / (spam.shape[0] + ham.shape[0])                  # fraction of pos examples

    print("model", model)
    return model

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    ##
    print("matrix:", matrix)
    ##
    model_pred = np.zeros(matrix.shape[0])

    log_phi_spam = np.sum(np.log(model['phi_spam']) * matrix, axis=1)   # using the hint to use logarithms
    log_phi_ham = np.sum(np.log(model['phi_ham']) * matrix, axis=1)     # using the hint to use logarithms
    phi = model['phi']

    ratio = np.exp(log_phi_ham + np.log(1 - phi) - log_phi_spam - np.log(phi))
    probs = 1 / (1 + ratio)

    model_pred[probs > 0.5] = 1

    return model_pred

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***

    top_five_indices = (-np.log(model['phi_spam'] / model['phi_ham'])).argsort()[:5]
    top_five_words = [word for word in dictionary.keys()]
    return [top_five_words[index] for index in top_five_indices]

    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

if __name__ == "__main__":
    main()
