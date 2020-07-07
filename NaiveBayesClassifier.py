import sys
import math
import time
import operator

train = []

text_pos = dict()
text_neg = dict()
vocab = set()
string_text = ""
ALPHA = 0.5


P_xk_pos_dict = dict()
P_xk_neg_dict = dict()

# record failed texts
# fail = []

def separate_doc_lab(arg1, arg2):
    # separate a whole line to [document,label]
    f1 = open(arg1, "r")
    f2 = open(arg2, "r")

    tot_doc = 0
    pos_doc = 0

    for l1 in f1.readlines():
        tot_doc += 1
        l1 = l1.rstrip('\n')
        sep = l1.split(',')
        if(sep[1]=="1"):
            pos_doc += 1
        train.append(sep)

    # P(pos) and P(neg)
    P_pos = pos_doc / tot_doc
    P_neg = (tot_doc - pos_doc) / tot_doc
    return P_pos,P_neg


def build_vocab(train_data):
    # a set of all words in train data
    for itr in range(len(train_data)):
        string_text = train_data[itr][0]
        voc = string_text.split(' ')
        for i in range(len(voc)):
            vocab.add(voc[i])

    vocab.remove('')

    # remove neutral words
    
    vocab.remove('the')
    vocab.remove('and')
    vocab.remove('a')
    vocab.remove('to')
    vocab.remove('i')
    vocab.remove('it')
    vocab.remove('is')
    vocab.remove('game')
    vocab.remove('of')
    vocab.remove('you')
    vocab.remove('have')
    vocab.remove('on')
    vocab.remove('as')
    vocab.remove('t')
    vocab.remove('was')
    vocab.remove('be')
    vocab.remove('my')
    vocab.remove('there')
    vocab.remove('or')
    vocab.remove('get')
    vocab.remove('at')
    vocab.remove('if')
    vocab.remove('your')
    vocab.remove('one')
    vocab.remove('this')
    vocab.remove('that')
    vocab.remove('in')
    vocab.remove('for')
    vocab.remove('but')
    vocab.remove('are')
    vocab.remove('s')
    vocab.remove('they')
    vocab.remove('with')
    vocab.remove('so')
    




def learning(data):
    n_forNeg = 0 
    n_forPos = 0 
    for itr in range(len(data)):

        if(data[itr][1]=="1"): # pos
            #### prepare text_pos
            # l_temp is the text of this line
            l_temp = data[itr][0].split(' ')
            l_temp.pop(len(l_temp)-1)
            # text_pos is a dict of words from pos_doc,
            # {word(str):count(int)}
            # put the words in this text into text_pos subset
            for ele in l_temp:
                n_forPos += 1
                try:
                    text_pos[ele] += 1
                except KeyError:
                    text_pos[ele] = 1
            
        if(data[itr][1]=="0"): # neg
            #### prepare text_neg
            l_temp2 = data[itr][0].split(' ')
            l_temp2.pop(len(l_temp2)-1)
            for ele2 in l_temp2:
                n_forNeg += 1
                try:
                    text_neg[ele2] += 1
                except KeyError:
                    text_neg[ele2] = 1
    
    ### remove neutral words
    text_pos.pop('the')
    text_pos.pop('and')
    text_pos.pop('a')
    text_pos.pop('to')
    text_pos.pop('i')
    text_pos.pop('it')
    text_pos.pop('is')
    text_pos.pop('game')
    text_pos.pop('of')
    text_pos.pop('you')
    text_pos.pop('have')
    text_pos.pop('on')
    text_pos.pop('as')
    text_pos.pop('t')
    text_pos.pop('was')
    text_pos.pop('be')
    text_pos.pop('my')
    text_pos.pop('there')
    text_pos.pop('or')
    text_pos.pop('get')
    text_pos.pop('at')
    text_pos.pop('if')
    text_pos.pop('your')
    text_pos.pop('one')
    text_pos.pop('this')
    text_pos.pop('that')
    text_pos.pop('in')
    text_pos.pop('for')
    text_pos.pop('but')
    text_pos.pop('are')
    text_pos.pop('s')
    text_pos.pop('they')
    text_pos.pop('with')
    text_pos.pop('so')

    text_neg.pop('the')
    text_neg.pop('and')
    text_neg.pop('a')
    text_neg.pop('to')
    text_neg.pop('i')
    text_neg.pop('it')
    text_neg.pop('is')
    text_neg.pop('game')
    text_neg.pop('of')
    text_neg.pop('you')
    text_neg.pop('have')
    text_neg.pop('on')
    text_neg.pop('as')
    text_neg.pop('t')
    text_neg.pop('was')
    text_neg.pop('be')
    text_neg.pop('my')
    text_neg.pop('there')
    text_neg.pop('or')
    text_neg.pop('get')
    text_neg.pop('at')
    text_neg.pop('if')
    text_neg.pop('your')
    text_neg.pop('one')
    text_neg.pop('this')
    text_neg.pop('that')
    text_neg.pop('in')
    text_neg.pop('for')
    text_neg.pop('but')
    text_neg.pop('are')
    text_neg.pop('s')
    text_neg.pop('they')
    text_neg.pop('with')
    text_neg.pop('so')
    

    ### calculate conditional prob

    for xk in vocab:
        if xk in text_pos:

            # num of appearance of xk
            nk = text_pos[xk]
            #print("this word is in text_pos ",xk)
            #print("with occurrences: ",nk)
            P_xk_pos = nk / (n_forPos)
            #print("P_xk_pos IN ",P_xk_pos)
            
            P_xk_pos_dict[xk] = P_xk_pos
            
        if xk not in text_pos:
            # xk not in text_pos
            #print("this word is NOT in text_pos ",xk)
            alpha = ALPHA
            P_xk_pos = alpha / (n_forPos+alpha*len(vocab))
            #print("P_xk_pos NOT IN ",P_xk_pos)
            P_xk_pos_dict[xk] = P_xk_pos

        if xk in text_neg:
            # num of appearance of xk
            nk = text_neg[xk]
            #print("this word is in text_neg ",xk)
            #print("with occurrences: ",nk)
            P_xk_neg = nk / (n_forNeg)
            #print("P_xk_neg IN ",P_xk_neg)
            P_xk_neg_dict[xk] = P_xk_neg
        
        if xk not in text_neg:
            alpha = ALPHA
            P_xk_neg = alpha / (n_forNeg+alpha*len(vocab))
            P_xk_neg_dict[xk] = P_xk_neg
            # n is the total number of words
            # in the merged text_neg
                    
    

def predicting(lst, print_label):
    word_bag = lst[0].rstrip('\n').split(' ')
    result_pos = math.log(P_pos)
    result_neg = math.log(P_neg)

    # print("just P pos ",result_pos)
    # print("just P neg ",result_neg)
    not_included_words = 0
    for word in word_bag:
        if word not in vocab:
            not_included_words += 1
            nk = 1
            alpha = ALPHA
            P_xk_pos = (nk + alpha) / ((1+alpha)*len(vocab))
            P_xk_neg = (nk + alpha) / ((1+alpha)*len(vocab))
            P_xk_pos_dict[word] = P_xk_pos
            P_xk_neg_dict[word] = P_xk_neg

        result_pos += math.log(P_xk_pos_dict[word])
        result_neg += math.log(P_xk_neg_dict[word])

    # print("result_pos ",result_pos)
    # print("result_neg ",result_neg)
    # print("count of added words ",not_included_words)
    label = 2
    if(result_pos>result_neg):
        label = 1
    else:
        label = 0
    
    if(print_label==1):
        print(label)

    if(result_pos>result_neg):
        return 1
    else:
        return 0

        

def predict_accuracy(arg, print_label):
    num_tested = 0
    num_accurate = 0
    f = open(arg, "r")
    for l in f.readlines():
        # [doc,label]
        num_tested += 1
        data = l.rstrip('\n').split(',')
        predict_result = predicting(data, print_label)
        actual_result = data[1]
        if(int(predict_result)==int(actual_result)):
            num_accurate += 1
        # else:
            # fail.append(data)
    accuracy = num_accurate / num_tested
    return accuracy

        


#### main

arg1 = str(sys.argv[1])
arg2 = str(sys.argv[2])

# separating
P_pos, P_neg = separate_doc_lab(arg1,arg2)

# building vocab
build_vocab(train)

# learning
learn_start = time.time()
learning(train)
learn_end = time.time()

# Accuracy
train_acc = predict_accuracy(arg1,0)

# predicting
predict_start = time.time()
test_acc = predict_accuracy(arg2,1)
predict_end = time.time()

# calculate training and testing time
print("{} seconds (training)".format(int(round(learn_end-learn_start))))
print("{} seconds (labeling)".format(int(round(predict_end-predict_start))))
print("{:.3f} (training)".format(train_acc))
print("{:.3f} (testing)".format(test_acc))

'''
# to get the 10 most important feature for pos and neg
P_xk_pos_dict_sorted = sorted(P_xk_pos_dict.items(),key = operator.itemgetter(1),reverse = True)
print("positive: most important features")
for i in range(0,10):
    print(i," : ",P_xk_pos_dict_sorted[i])

P_xk_neg_dict_sorted = sorted(P_xk_neg_dict.items(),key = operator.itemgetter(1),reverse = True)
print("negative: most important features")
for i in range(0,10):
    print(i," : ",P_xk_neg_dict_sorted[i])
'''



