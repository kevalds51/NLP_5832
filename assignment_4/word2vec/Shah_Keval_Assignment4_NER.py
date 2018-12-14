import numpy
import  pickle as pickle

################## load model ######################

with open('arr_tags.pickle', 'rb') as handle:
    arr_tags = pickle.load( handle)


with open('dict_tags.pickle', 'rb') as handle:
    dict_tags = pickle.load( handle)


with open('arr_words.pickle', 'rb') as handle:
    arr_words = pickle.load( handle)


with open('emission_counts.pickle', 'rb') as handle:
    emission_counts = pickle.load( handle)


with open('transition_prob.pickle', 'rb') as handle:
    transition_prob = pickle.load( handle)

with open('pi_map.pickle', 'rb') as handle:
    pi_map = pickle.load( handle)

with open('pi.pickle', 'rb') as handle:
    pi = pickle.load( handle)

############# model ends here #############################



f = open('test_lines.txt', 'w')
for line in open("TestNER.txt"):
    inline = line.strip().split()
    if len(inline) >= 2:
        f.write(inline[1] + ' ')
    else:
        f.write("\n")
f.close()


count = 0

predicted_tag_array = []


################################# testing starts here #############################
#read train vocab to test vocab map
train_test_dict={}
with open('word_conv.pickle', 'rb') as handle:
    train_test_dict = pickle.load( handle)

for line in open("test_lines.txt", "r"):
    inline = line.strip().split()
    T = len(inline)
    viterbi_matrix = numpy.zeros((len(arr_tags), T))
    T2 = numpy.zeros((len(arr_tags), T),dtype=int)
    tag_array = [0 for x in range(T)]
    tag1 = 'O'
    s=0

    for s in range(len(arr_tags)) :
        word_index = arr_words.index(train_test_dict[inline[0]])
        emission_prob = 0 # non zero
        if emission_counts[s][word_index] > 0:
            emission_prob = emission_counts[s][word_index] / dict_tags[arr_tags[s]]
        viterbi_matrix[s][0] = pi[s] * emission_prob
        T2[s][0] = 0

    t = 1
    s = 0
    s1 = 0
    for t in range(1,T):
        for s in range(len(arr_tags)):
            word_index = arr_words.index(train_test_dict[inline[t]])
            emission_prob = 0 # non zero
            if emission_counts[s][word_index] > 0:
                emission_prob = emission_counts[s][word_index] / dict_tags[arr_tags[s]]

            for s1 in range(len(arr_tags)):
                temp = viterbi_matrix[s1][t-1] * transition_prob[s1][s]*emission_prob
                if temp > viterbi_matrix[s][t]:
                    viterbi_matrix[s][t] = temp
                    T2[s][t] = s1

        #tag_array.append(arr_tags[numpy.nanargmax(viterbi_matrix[:,t])])

    ### output the sequence
    last_state = numpy.nanargmax(viterbi_matrix[:,T-1])
    tag_array[T-1] = arr_tags[last_state]
    for t in range(T-1,0,-1):
        last_state = T2[last_state][t]
        tag_array[t-1] = arr_tags[last_state]
    predicted_tag_array.extend(tag_array)

    count += 1
    print (count)

print (predicted_tag_array)
f = open("outputNER.txt", 'w')
i = 0
for line in open("TestNER.txt", 'r') :
    inline = line.strip().split()
    if len(inline) < 2:
        f.write('\n')
        continue
    f.write(inline[0] + '\t' + inline[1] + '\t' + predicted_tag_array[i] + '\n')
    i += 1

