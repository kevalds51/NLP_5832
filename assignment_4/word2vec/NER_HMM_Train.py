import numpy
import  pickle
dict_words = {}
dict_tags = {}
no_of_sequence=1
ocount = 1
icount = 0
bcount = 0
first=0

for line in open("gene-trainF17.txt", "r"):
    inline = line.strip().split()
    if len(inline) != 3:
        no_of_sequence += 1
        first = 1
        continue
    tag = inline[2]
    if first == 1 :
        if tag == "O":
            ocount += 1
        else :
            if tag == "I":
                icount += 1
            else:
                bcount += 1
        first = 0

    if tag in dict_tags.keys():
        dict_tags[tag] += 1
    else:
        dict_tags[tag] = 1
    word = inline[1]
    if word in dict_words.keys():
        dict_words[word] += 1
    else:
        dict_words[word] = 1

pi_map = {}
pi_map['O'] = ocount*1.0/no_of_sequence
pi_map['I'] = icount*1.0/no_of_sequence
pi_map['B'] = bcount*1.0/no_of_sequence

pi=[]
arr_tags = []

for key in dict_tags.keys():
    arr_tags.append(key)
    pi.append(pi_map[key])

arr_words = []
for key in dict_words.keys():
    arr_words.append(key)

### three unknowns to handle words not seen in training data
arr_words.append('unknown_aplha')
arr_words.append('unknown_numeric')
arr_words.append('unknown_alphanumeric')
arr_words.append('unknown_special')


emission_counts = numpy.zeros((len(arr_tags), len(arr_words)))

i = 0

for line in open("gene-trainF17.txt", "r"):
    inline = line.strip().split()
    #print (inline)
    if len(inline) < 3:
        continue
    word = inline[1]
    tag = inline[2]

    label='unknown_special'
    if word.isalpha():
         label = 'unknown_aplha'
    elif word.isdigit():
         label = 'unknown_numeric'
    elif word.isalnum():
         label = 'unknown_alphanumeric'

    emission_counts[arr_tags.index(tag)][arr_words.index(word)] += 1
    emission_counts[arr_tags.index(tag)][arr_words.index(label)] += 1 #for unknowns


transition_prob = numpy.zeros((len(arr_tags), len(arr_tags)))
dict_transition = {}
tag1 = 'O'

for line in open("gene-trainF17.txt","r"):
    inline = line.strip().split()

    if len(inline) != 3:
        continue
    tag2 = inline[2]
    if tag2+"/"+tag1 in dict_transition.keys():
        dict_transition[tag2+"/"+tag1] += 1
    else:
        dict_transition[tag2 + "/" + tag1] = 1
    tag1 = tag2
print (dict_transition)
for key in dict_transition.keys():
    inline = key.strip().split("/")
    tag1 = arr_tags.index(inline[1])
    tag2 = arr_tags.index(inline[0])
    transition_prob[tag1][tag2] = dict_transition[key]

for i in range(len(arr_tags)):
    transition_prob[i] /= transition_prob[i].sum()
final_tag_array = []



#print (arr_tags)
#print (arr_words)
print ("Emission counts",emission_counts)
print ("Transition Prob",transition_prob)

with open('emission_counts.pickle', 'wb') as handle:
    pickle.dump(emission_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('transition_prob.pickle', 'wb') as handle:
    pickle.dump(transition_prob, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('dict_tags.pickle', 'wb') as handle:
    pickle.dump(dict_tags, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('arr_tags.pickle', 'wb') as handle:
    pickle.dump(arr_tags, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('arr_words.pickle', 'wb') as handle:
    pickle.dump(arr_words, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pi_map.pickle', 'wb') as handle:
    pickle.dump(pi_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pi.pickle', 'wb') as handle:
    pickle.dump(pi, handle, protocol=pickle.HIGHEST_PROTOCOL)