import numpy as np
from sklearn import tree
import sys

def calculate_accuracy(predictions, true_labels):
    assert len(predictions) == len(true_labels)
    correct_preds = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predictions[i]:
            correct_preds += 1
    return correct_preds / len(true_labels) * 100

def generate_iob_tags(class_definitions):
    tokens = []
    for value in class_definitions[1].values():
        words = value.split(" ")
        for j in range(len(words)):
            if j == 0:
                tokens.append((words[j], 'B'))
            else:
                tokens.append((words[j], 'I'))
    return tokens

def gen_annotations(token, iob_tags):
    token_val = token.lower()
    uppercase = 0 
    if token.isupper():
        uppercase = 1
    else:
        0
    startscapital = 0 
    if token[0].isupper():
        startscapital =1 
    else:
        0
    length = len(token)
    numbers = 0 
    if token.isdigit():
        numbers = 1 
    else:
        0
    endsnumeral = 0
    if token[-1].isnumeric():
        endsnumeral = 1 
    else:
        0
    lengthoverthree = 0 
    if len(token) > 3:
        lengthoverthree = 1
    else:
        0
    endscapital = 0
    if token[-1].isupper():
        endscapital = 1
    else:
        0
    annotation = 'O'
    for i in range(len(iob_tags)):
        if token == iob_tags[i][0]:
            annotation = iob_tags[i][1]
            break

    if token.lower() not in tokenization_dict:
        tokenization_dict[token.lower()] = len(tokenization_dict) + 1

    return_vals = (token_val, uppercase, startscapital, length, numbers,
            endsnumeral, lengthoverthree, endscapital, annotation)
    return return_vals

def parse(sequence):
    sequence_annots = {}
    sequence = sequence.split("<class")[1]
    elements = sequence.strip().split()
    tags = []
    keys = []
    for content in elements:
        if "instructor" in content:
            break
        tags_list = content.split("=")
        if len(tags_list) == 2:
            value = []
            if tags_list[0] not in tags:
                tags.append(tags_list[0])
                value.append(tags_list[1])
            keys.append(" ".join(value))
        elif len(tags_list) == 1:
            value.append(tags_list[0])
            keys[-1] = " ".join(value)

    for t in range(len(tags)):
        if tags[t] == "id":
            sequence_annots["id"] = keys[t]
        if tags[t] == "name":
            sequence_annots["name"] = keys[t]
    return sequence_annots

def preprocess(data):
    separated_data_with_class = []
    current_sequence = []
    for line in data:
        if not line.isspace():
            current_sequence.append(line)
        else:
            text = "".join(current_sequence)
            text = text.replace("\n", " ").strip()
            text = text.replace("(", "").replace(")", "")
            if "<class" not in text:
                continue
            text = text.replace(">", "")
            separated_data_with_class.append(text)
            current_sequence = []

    classes = []
    for i, seq in enumerate(separated_data_with_class):
        class_definitions = parse(seq)
        if class_definitions != {}:
            classes.append((i, class_definitions))
    X_data = []
    for i in range(len(classes)):
        iob_tags = generate_iob_tags(classes[i])
        text_tokens = separated_data_with_class[classes[i][0]].split(".")[0].split()
        for token in text_tokens:
            annotation = gen_annotations(token, iob_tags)
            X_data.append(annotation)
    X_data = np.asarray(X_data)
    return X_data, separated_data_with_class

def input_preprocessing(X):
    X = np.apply_along_axis(lambda column: np.vectorize({value: idx for idx, value in enumerate(np.unique(column))}.get)(column), axis=0, arr=X)
    return X

if __name__ == '__main__':
    tokenization_dict = {}

    training_filename = sys.argv[1]
    testing_filename = sys.argv[2]
    train_file = open('NLU.train', 'r')
    train_data = train_file.readlines()

    X_data_train,sequences_train = preprocess(train_data)

    X_train = X_data_train[:, :-1]
    Y_train = X_data_train[:, -1]

    X_train = input_preprocessing(X_train)

    model_train = tree.DecisionTreeClassifier()
    model_train.fit(X_train, Y_train)

    predictions_train = model_train.predict(X_train)
    print("Training accuracy:", calculate_accuracy(predictions_train, Y_train))

    test_dataset = []
    test_file = open('NLU.test','r')
    test_data = test_file.readlines()

    X_data_test,sequences_test = preprocess(test_data)

    for i, sentence in enumerate(sequences_test):
        print('input: {}'.format(sentence))
        ex_test = parse(sentence)
        if ex_test != {}:
            iob_test = generate_iob_tags((i, ex_test))
            text_test = sentence.split(".")[0].split()

            formatted_tokens = []
            for j, token in enumerate(text_test):
                an_test = gen_annotations(token, iob_test)
                formatted_tokens.append(token+'/'+an_test[0])

            formatted_sentence = " ".join(formatted_tokens)
            print('output: {}'.format(formatted_sentence + '\n'))

    X_test = X_data_test[:, :-1]
    Y_test = X_data_test[:, -1]

    X_test = input_preprocessing(X_test)

    predictions_test = model_train.predict(X_test)

    print("Testing accuracy:", calculate_accuracy(predictions_test, Y_test))