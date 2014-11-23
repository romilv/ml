import json
import mlpy
import numpy
import math
import itertools
import heapq


# AMAZON TO AMAZON
SOLVER = 'l2r_l2loss_svc' # best l2r_l2loss_svc
C = 0.2 # default is 1

with open('./stats/task5_stats.dat', 'a') as f:
    f.write('\n\n--------')
    f.write('\nSOLVER ' + SOLVER)
    f.write('\nC ' + str(C))
    f.write('\nSIZE 1500')
f.close()

x_file_name = './data/task2_tfidf2d_list1500.json'
x_file_data = open(x_file_name).read()
x = numpy.array(json.loads(x_file_data))

y_file_name = './data/task2_y1500.json'
#Get number of labels
numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json')))
#Generate y as a 2D array where we indicate presence/absence of each label
#y is column major
#Also build the correlation matrix
arr = [0]*len(x)
y = numpy.array([list(arr) for _ in xrange(numLabels)])
corr_y = numpy.array([list([0.0]*numLabels) for _ in xrange(numLabels)])
y_open = open(y_file_name)
y_data = numpy.array(json.loads(y_open.read()))
for i,label_list in enumerate(y_data):
    for label in label_list:
        y[label][i]=1
    for item in itertools.permutations(label_list,2):
        corr_y[item[0]][item[1]]+=1.0
    pass

# Normalize the correlation matrix
for idx,label in enumerate(corr_y):
    total = sum(label)
    if total==0:
        continue
    corr_y[idx]*=1.0/total

TRAIN_PERCENT = 0.7
split_index = int(math.floor(TRAIN_PERCENT * len(x)))

#Train model for all labels
models = []
for label in y:
    temp = mlpy.LibLinear(solver_type=SOLVER, C=C)
    temp.learn(x[:split_index],label[:split_index])
    models.append(temp)
    pass

def error(pred_list, actual_set, n):
    # n >= 1
    correct= 0.0
    if len(pred_list) != 0:
        for elem in pred_list:
            if elem  in actual_set:
                correct += 1.0
        # ~ fraction match in actual labels time ~ fraction overmatched in predicted labels
        return 1-(correct/len(actual_set)) * math.pow((correct/len(pred_list)), 1.0/n)
    else:
        return 1
    
def evaluate_model(test_type, start, end):
    heap_err, cor_err, nocor_err, pred_err = 0, 0, 0, 0
    for i in range(start,end):
        probabilities = [(1-model.pred_probability(x[i])[0], idx) for idx,model in enumerate(models)] 
        
        predictions = [idx for idx,model in enumerate(models) if model.pred(x[i])==1 ]
        probabilities.sort(reverse=True)
        probabilities =  probabilities[:10] # consider 10 with highest Pr
        
        augment_probs = list()
        for tup1 in probabilities:
            val = 0
            for tup2 in probabilities:
                if tup1[1] != tup2[1]:
                    # not considering correlation matrix
                    val += tup1[0]*tup2[0] 

            augment_probs.append((val,tup1[1]))

        augment_probs2 = list()
        for tup1 in probabilities:
            val = 0
            for tup2 in probabilities:
                if tup1[1] != tup2[1]:
                    # with correlation matrix
                    val += tup1[0]*tup2[0]*(corr_y[tup1[1]][tup2[1]]+corr_y[tup2[1]][tup1[1]])
            augment_probs2.append((val,tup1[1]))

        augment_probs2.sort(reverse=True)
        augment_probs.sort(reverse=True)

        # considering tuples from correlation matrix and choosing top K
        h = []
        pairs = itertools.combinations(probabilities,2)
        for a,b in pairs:
            pr = a[0]*b[0]*(corr_y[a[1]][b[1]]+corr_y[b[1]][a[1]])
            key = (a[1], b[1])
            heapq.heappush(h, (pr, key))
        h1 = heapq.nlargest(5, h)
        final_h = set()
        for elem in h1: # distinct
            final_h.add(elem[1][0])
            final_h.add(elem[1][1])
        final_h = list(final_h)

        # split at max probability drop
        max_diff = float('-inf')
        split_idx = -1
        for j in range(1, len(augment_probs)):
            diff = augment_probs[j-1][0] - augment_probs[j][0]
            if diff > max_diff:
                max_diff = diff
                split_idx = j
        augment_probs = augment_probs[:split_idx]

        max_diff = float('-inf')
        split_idx = -1
        for j in range(1, len(augment_probs2)):
            diff = augment_probs2[j-1][0] - augment_probs2[j][0]
            if diff > max_diff:
                max_diff = diff
                split_idx = j
        augment_probs2 = augment_probs2[:split_idx]
        # print "after split", augment_probs
        cor_q = [val[1] for val in augment_probs]
        # print "with cor", q
        nocor_q = [val[1] for val in augment_probs2]
        # print "no cor", q
        actual_labels = set(y_data[i])
        # print "actual", actual_labels, "\n"

        n = 3.0
        heap_err += error(final_h, actual_labels, n)
        cor_err += error(cor_q, actual_labels, n)
        nocor_err += error(nocor_q, actual_labels, n)
        pred_err += error(predictions, actual_labels, n)

    with open('./stats/task5_stats.dat', 'a') as f:
        f.write("\nerror type " + test_type)
        f.write("\nerror n " +  str(n))
        f.write('\nheap ' + str(heap_err/(end-start)))
        f.write("\ncorr " + str(cor_err/(end-start)))
        f.write("\nnocor " + str(nocor_err/(end-start)))
        f.write("\npred_err " + str(pred_err/(end-start)))
        f.write("\n------")
    f.close()
    # end evaluate_model

evaluate_model('TRAIN', 0, split_index)
evaluate_model('TEST', split_index, len(x))
