import numpy as np
import sys

def tokenizeUniqueWords(corpus): 
    unique = [] 
    unique.append(corpus[0])
    for i in corpus:
        if not i in unique:
            unique.append(i)
    tokenizeddict = {}
    for i in range(len(unique)):
        tokenizeddict[unique[i]] = i
    return tokenizeddict

def getUniqueElements(corpus):
    unique = [] 
    unique.append(corpus[0])  
    for i in corpus:
        if not i in unique:
            unique.append(i)
    return unique

trainfile = open('POS.train', 'r')
testfile = open('POS.test', 'r')

file = trainfile
lines = file.readlines()
lines_new = []
for line in lines:
    lines_new.append(line.split())
wordsandtags = []
for i in range(len(lines_new)):
    tags = []
    words = []
    for token in lines_new[i]:
        words.append(token.split('/')[0])
        try:
            tags.append(token.split('/')[1])
        except IndexError:
            tags.append('NUM')
    wordsandtags.append(words)
    wordsandtags.append(tags)
dict_word_count = {}
dict_word_tag_count = {}
tag_bigrams = {}
dict_tag_count = {}
for i in range(0,len(wordsandtags),2):
    for j in range(0,len(wordsandtags[i])):
        word = wordsandtags[i][j]
        tag = wordsandtags[i+1][j]
        try:
            dict_tag_count[tag.lower()] += 1
        except KeyError:
            dict_tag_count[tag.lower()] = 1
        if j == 0:
            last_tag = '__'
        else:
            last_tag = wordsandtags[i+1][j-1]
        
        try:
            dict_word_count[word.lower()] += 1
        except KeyError:
            dict_word_count[word.lower()] = 1
        
        try:
            dict_word_tag_count [str(word.lower()) + ' ,' + str(tag.lower())] += 1
        except KeyError:
            dict_word_tag_count [str(word.lower()) + ' ,' + str(tag.lower())] = 1
        
        try:
            tag_bigrams[str(tag.lower()) + ' ,' + str(last_tag.lower())] += 1
        except KeyError:
            tag_bigrams[str(tag.lower()) + ' ,' + str(last_tag.lower())] = 1

tags = list(dict_tag_count.keys())

print('Large training complete')
test_file = testfile
test_lines = test_file.readlines()
tags_grand = []
words_grand = []
for line in lines:
    wordsandtags = line.split()
    words = []
    tags = []
    for wordtag in wordsandtags:
        wordtag = wordtag.split('/')
        words.append(wordtag[0])
        try:
            tags.append(wordtag[1])
        except IndexError:
            tags.append('NUM')
    words_grand.append(words)
    tags_grand.append(tags)
tags_grand_linear = []
for i in range(len(tags_grand)):
    for j in range(len(tags_grand[i])):
        tags_grand_linear.append(tags_grand[i][j].lower())
tags_grand_linear = getUniqueElements(tags_grand_linear)
tags = tags_grand_linear
for i in range(len(tags)):
    tags[i] = tags[i].lower()
tags_unique_tokenized_dict = tokenizeUniqueWords(tags)
sentencewise_POS = []
for i in range(len(words_grand)):
    scores = np.zeros((len(tags), len(words_grand[i])))
    backpointer_arr = []
    for k in range(len(words_grand[i])):
        backpointer_arr.append(' ')
    POS_list = []
    for j in range(len(words_grand[i])):
        word = words_grand[i][j]
        if (j == 0):
            start_of_sentence = True
        else:
            start_of_sentence = False
        keys_dict_word_tag_count = dict_word_tag_count.keys()
        possible_tags = []
        for key in keys_dict_word_tag_count:
            if key.startswith(word.lower() + ' ,'):
                wordlen = len(word.lower()) + 2
                tag_extracted =  key[wordlen:]
                possible_tags.append(tag_extracted)
        if (start_of_sentence):
            dict = {}
            for tag in possible_tags:
                try:
                    score = dict_word_tag_count[word.lower() + ' ,' + tag.lower()] / dict_word_count[word.lower()] * tag_bigrams[str(tag.lower()) + ' ,' + str('__')] / dict_tag_count[tag.lower()]
                except KeyError:
                    score = 0
                dict[tag.lower()] = score
                scores[tags_unique_tokenized_dict[tag.lower()], j] = dict[tag.lower()]
        else:
            dict = {}
            for tag in possible_tags:
                prev_max_dict = {}
                for prev_index in range(scores[:,j-1].shape[0]):
                    val_list = list(tags_unique_tokenized_dict.values())
                    key_list = list(tags_unique_tokenized_dict.keys())
                    position = val_list.index(int(prev_index))
                    prev_tag = key_list[position].lower()
                    try:
                        prev_max_dict[prev_tag] = scores[int(prev_index),int(j-1)] * tag_bigrams[prev_tag + ' ,' + tag.lower()]
                    except KeyError:
                        prev_max_dict[prev_tag] = 0
                val_list_max = np.array(list(prev_max_dict.values()))
                key_list_max = list(prev_max_dict.keys())
                index_work = np.where(val_list_max == val_list_max.max())[0][0]
                backpointer = key_list_max[index_work]
                backpointer_arr[j-1] = backpointer
                score = dict_word_tag_count[word.lower() + ' ,' + tag.lower()] * val_list_max.max()
                scores[tags_unique_tokenized_dict[tag], j] = score
        if j == len(words_grand[i]):
            last_tag_index = np.where(scores[:,j] == scores[:,j].max())[0]
            val_list = np.array(tags_unique_tokenized_dict.values())
            key_list = list(tags_unique_tokenized_dict.keys())
            position = val_list.index(int(last_tag_index))            
            last_tag = key_list(position)
            backpointer_arr[j] = last_tag
    for m in range(len(backpointer_arr)):
        POS_list.append(backpointer_arr[m])
    sentencewise_POS.append(POS_list)

for i in range(len(tags_grand)):
    for j in range(len(tags_grand[i])):
        tags_grand[i][j] = tags_grand[i][j].lower()

accuracy = []
for i in range(len(sentencewise_POS)):
    for j in range(len(sentencewise_POS[i])):
        if sentencewise_POS[i][j] == tags_grand[i][j]:
            accuracy.append(1)
        else:
            accuracy.append(0)
accuracy = np.array(accuracy)
accuracy = np.mean(accuracy)
print('Accuracy with training on {} dataset is {}%'.format('POS.train', accuracy*100))