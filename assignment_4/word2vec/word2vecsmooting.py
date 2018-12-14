import gensim
import  pickle

model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True,
                                                        limit=700000)
#model1 = gensim.models.KeyedVectors.load_word2vec_format('./g2wv', binary=False)
############ get all training words ##############
arr_words = []

for line in open("gene-trainF17.txt", "r"):
    inline = line.strip().split()
    if len(inline) != 3:
        continue

    word = inline[1]
    arr_words.append(word)


#test vocabulary
test_arr_words = []
for line in open("TestNER.txt", 'r') :
    inline = line.strip().split()
    if len(inline) < 2:
        continue
    word = inline[1]
    test_arr_words.append(word)


### check overlap of vocabulary
vocab_train  = set(arr_words)
vocab_test = set(test_arr_words)

w2vecmodel=[]
for  w in vocab_train:
    if w in model.wv.vocab:
        w2vecmodel.append(w)

print(len(w2vecmodel))
### map test words to train words
train_test_dict={}
count=0
c_w=0


for word in vocab_test:
    if train_test_dict.has_key(word):
        continue
    elif word in vocab_train:
        train_test_dict[word] = word
    elif word in model.wv.vocab:
        sim = -1
        c_w += c_w
        for w in w2vecmodel:
            word_sim = model.similarity(w,word)
            if word_sim >= sim:
                sim = word_sim
                train_test_dict[word] = w
    else:
        count += 1

        label='unknown_special'
        if word.isalpha():
           label = 'unknown_aplha'
        elif word.isdigit():
            label = 'unknown_numeric'
        elif word.isalnum():
            label = 'unknown_alphanumeric'
        train_test_dict[word] = label
        print(word+" : "+label)

c=0
for w in vocab_test:
    if not train_test_dict.has_key(w):
        c+=1
print("missing "+str(c))

print("unknowns "+ str(count))
print("wv "+ str(c_w))
#print(train_test_dict)
with open('word_conv.pickle', 'wb') as handle:
    pickle.dump(train_test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('word_conv.pickle', 'rb') as handle:
    l_train_test_dict = pickle.load( handle)
    print( l_train_test_dict==train_test_dict)


