from sklearn import tree
import numpy as np
from collections import Counter
import sys

def tokenizeUniqueWords(corpus): 
    '''
    This function makes a list of the unique tokens present in a set of corpus and generates a tokenization scheme for them. This function is called in both train() and test()

    Input:
    'corpus': This contains the corpus that we want to tokenize uniquely

    Output:
    'tokenizeddict': This is the tokenization dictionary that contains all unique words present in the corpus and an integer 'token' for them 
    '''
    unique = [] 
    unique.append(corpus[0])
    for i in corpus:
        if not i in unique:
            unique.append(i)
    tokenizeddict = {}
    for i in range(len(unique)):
        tokenizeddict[unique[i]] = i
    return tokenizeddict

def pre_process(tokens, tokenized_dictionary, list): 
    '''
    This function Extracts features from the period instance.

    Inputs: 
    'list' contains the indices that have period in the tokens
    'tokenized_dictionary' contains keys that are unique words in the tokens with values that is their conversion to an integer token 
    'tokens' contains the list of tokens from the input deck

    Outputs:
    'left_list': list of words that are present to the left of a period
    'right_list': list of words that are present to the right of a period
    'L3_list': list of binary 1s and 0s to represent whether the left word is over 3 characters long
    'lUlist': list of binary 1s and 0s to represent whether the left word is capitalized or not
    'rUlist': list of binary 1s and 0s to represent whether the right word is capitalized or not
    'indices': indices where periods occur without empty tokens
    '''
    left_list = []
    right_list = []
    L3list = []
    R3list = []
    lUlist = []
    rUlist = []
    indices = []
    list_of_punctuations = ['"', ')', '(']
    for i in list:
        left = ""
        right = ""
        if len(tokens[i]) != 0:
            current = tokens[i]
            while(current[-1] =='.' and len(current) != 0):
                current = current[:-1]
                if len(current) == 0:
                    break
            L3 = 0
            R3 = 0
            lU = 0
            rU = 0
            if len(current) > 1:
                left = current
            else:
                left = tokens[i-1]
            if (i+1 < len(tokens)):
                right_index = i+1
                while tokens[right_index] in list_of_punctuations:
                    right_index += 1
                right = tokens[right_index]
            if (left != "" and len(left)>3):
                L3 = 1
                if (left[0].isupper()):
                    lU = 1
            if (right != ""):
                if (right[0].isupper()):
                    rU = 1
            if (len(left) != 0):
                while left[-1] == '.':
                    if (len(left) != 0):
                        left = left[:-1]
                    else:
                        break
            if (len(right) != 0):
                while right[-1] == '.':
                    if len(right)!= 0:
                        right = right[:-1]
                    else:
                        break
            if (len(right)>3):
                R3 = 1
            R3list.append(R3)
            left_list.append(tokenized_dictionary[left.lower()])
            right_list.append(tokenized_dictionary[right.lower()])
            L3list.append(L3)
            lUlist.append(lU)
            rUlist.append(rU)
            indices.append(i)
    return left_list, right_list, L3list, lUlist, rUlist, R3list, indices

def train(tokens, labels):
    '''
    This function creates a classifier tree and trains it on the input data

    Inputs:
    'tokens': un-processed corpus containing periods
    'labels': un-processed labels labelling periods as EOS or NEOS. For simplicity, the function converts them to 1s for EOS and 0s for NEOS

    Outputs:
    'clf': The classifier Tree trained on the input dataset
    'tokenized_dictionary': The tokenization dictionary used on this dataset so we can use the same dictionary for any and all predictions
    '''
    period_index_list = []
    for throwawy,i in enumerate(tokens):
        if i == '.' or i[-1] == '.':
            period_index_list.append(throwawy)
    lowercasetokens = []
    for i in range(len(tokens)):
        if len(tokens[i]) != 0 and tokens[i] [-1] == '.':
            while tokens[i][-1] == '.':
                if len(tokens[i]) != 0:
                    tokens[i] = tokens[i][:-1]
                    if len(tokens[i]) == 0:
                        break
                else:
                    break
        lowercasetokens.append(tokens[i].lower())
    tokenized_dictionary = tokenizeUniqueWords(lowercasetokens)
    left_list, right_list, L3list, lUlist, rUlist, R3list, indices = pre_process(tokens, tokenized_dictionary, period_index_list)
    labelsperiod = []
    for i in indices:    
        labelsperiod.append(labels[i])
    for i in range(len(lowercasetokens)):
        if len(lowercasetokens[i]) != 0:
            while lowercasetokens[i][-1] == '.':
                lowercasetokens[i] = lowercasetokens[i][:-1]
                if len(lowercasetokens[i]) ==0:
                    break
    for i in range(len(tokens)):
        word = tokens[i].lower()
        tokens[i] = tokenized_dictionary[word]

    #Compute frequencies
    frequencies = Counter(tokens)

    keys = list(frequencies.keys())
    values = np.array(list(frequencies.values()))
    values = values/len(tokens)
    values = list(values)
    frequency_dict = {}
    for key in keys:
        for value in values:
            frequency_dict[key] = value
            values.remove(value)
            break
    frequency_l = []
    frequency_r = []
    for i in range(len(left_list)):
        left = left_list[i]
        freq_l = frequency_dict[left]
        frequency_l.append(freq_l)
        right = right_list[i]
        freq_r = frequency_dict[right]
        frequency_r.append(freq_r)
    X = np.column_stack((left_list, right_list, L3list, lUlist, rUlist, R3list, frequency_l, frequency_r))
    for i in range(len(labelsperiod)):
        if labelsperiod[i] == 'NEOS':
            labelsperiod[i] = 0
        else:
            labelsperiod[i] = 1
    Y = np.array(labelsperiod)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X,Y)
    return clf, tokenized_dictionary

def test(tokens, labels, tokenized_dictionary, clf):
    '''
    Inputs:

    'tokens': un-processed testing corpus containing periods
    'labels': un-processed testing labels labelling periods as EOS or NEOS. For simplicity, the function converts them to 1s for EOS and 0s for NEOS
    'tokenized_dictionary': The tokenization dictionary used for training this particular classifier tree
    'clf': The classifier tree we are testing

    Output:
    'accuracy': The accuracy of the classifier tree on the provided test dataset
    '''
    period_index_list = []
    for throwawy,i in enumerate(tokens):
        if i == '.' or i[-1] == '.':
            period_index_list.append(throwawy)
    lowercasetokens = []
    for i in range(len(tokens)):
        if len(tokens[i]) != 0 and tokens[i] [-1] == '.':
            while tokens[i][-1] == '.':
                if len(tokens[i]) != 0:
                    tokens[i] = tokens[i][:-1]
                    if len(tokens[i]) == 0:
                        break
                else:
                    break
        lowercasetokens.append(tokens[i].lower())
    left_list, right_list, L3list, lUlist, rUlist, R3list, indices = pre_process(tokens, tokenized_dictionary, period_index_list)
    labelsperiod = []
    for i in indices:    
        labelsperiod.append(labels[i])
    for i in range(len(lowercasetokens)):
        if len(lowercasetokens[i]) != 0:
            while lowercasetokens[i][-1] == '.':
                lowercasetokens[i] = lowercasetokens[i][:-1]
                if len(lowercasetokens[i]) ==0:
                    break
    for i in range(len(tokens)):
        word = tokens[i].lower()
        tokens[i] = tokenized_dictionary[word]

    #Compute frequencies
    frequencies = Counter(tokens)
    keys = list(frequencies.keys())
    values = np.array(list(frequencies.values()))
    values = values/len(tokens)
    values = list(values)
    frequency_dict = {}
    for key in keys:
        for value in values:
            frequency_dict[key] = value
            values.remove(value)
            break
    frequency_l = []
    frequency_r = []
    for i in range(len(left_list)):
        left = left_list[i]
        freq_l = frequency_dict[left]
        frequency_l.append(freq_l)
        right = right_list[i]
        freq_r = frequency_dict[right]
        frequency_r.append(freq_r)
    X = np.column_stack((left_list, right_list, L3list, lUlist, rUlist, R3list, frequency_l, frequency_r))
    for i in range(len(labelsperiod)):
        if labelsperiod[i] == 'NEOS':
            labelsperiod[i] = 0
        else:
            labelsperiod[i] = 1
    Y = np.array(labelsperiod)
    prediction = clf.predict(X)
    diff = prediction - Y
    diff = np.array(diff)
    wrong_preds = np.where(diff != 0)
    accuracy = (1 - len(wrong_preds)/len(prediction))*100
    return accuracy

if __name__ == '__main__':
    '''
    Main function that opens the first filename provided, trains the classifier on that file and then tests it on the second file provided. File names are expected in the CLI. 
    Sample input for CLI: python3 SBD.py SBD.train SBD.test
    
    '''
    #Open Training data
    train_data = open(sys.argv[1],'r')
    lines = train_data.readlines()
    #extract tokens and labels from the training raw data
    tokens_train = []
    labels_train = []
    for line in lines:
        line = line.split()
        tokens_train.append(line[1])
        labels_train.append(line[2])
    #Train a tree
    clf, tokenized_dictionary = train(tokens_train, labels_train)

    #Open testing Data
    test_data = open(sys.argv[2], 'r')
    lines_test = test_data.readlines()
    #Extract tokens and labels from the testing raw data
    tokens_test = []
    labels_test = []
    for line in lines:
        line = line.split()
        tokens_test.append(line[1])
        labels_test.append(line[2])
    accuracy = test(tokens_test, labels_test, tokenized_dictionary, clf)
    print('The Accuracy for using these features is: {}%'.format(round(accuracy,2)))