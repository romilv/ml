import json
import mlpy
import numpy
import math
import itertools
import heapq


# AMAZON TO AMAZON
SOLVER = 'l2r_l2loss_svc' # best l2r_l2loss_svc
C = 1 # default is 1

with open('./stats/task6_stats.dat', 'a') as f:
    f.write('\n\n--------')
    f.write('\nSOLVER ' + SOLVER)
    f.write('\nC ' + str(C))
    f.write('\nSIZE 3000')
f.close()

x_file_name = './data/task2_tfidf2d_list3000.json'
x_file_data = open(x_file_name).read()
x = numpy.array(json.loads(x_file_data))

y_file_name = './data/task2_y3000.json'
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

# smoothing by 1
corr_y += 1

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
    
def evaluate_model(test_type, start, end, alpha):
    heap_err, pred_err = 0, 0
    for i in range(start,end):
        probabilities = [(1-model.pred_probability(x[i])[0], idx) for idx,model in enumerate(models)] 
        
        predictions = [idx for idx,model in enumerate(models) if model.pred(x[i])==1 ]
        probabilities.sort(reverse=True)
        probabilities =  probabilities[:10] # consider 10 with highest Pr

        # considering tuples from correlation matrix and choosing top K=5
        # heap
        h = []
        pairs = itertools.combinations(probabilities,2)
        for a,b in pairs:
            p_a_b = math.sqrt(corr_y[a[1]][b[1]] * corr_y[b[1]][a[1]])
            p_a_b = math.pow(p_a_b, alpha)

            pr = a[0]*b[0]*p_a_b
            key = (a[1], b[1])
            heapq.heappush(h, (pr, key))
        h1 = heapq.nlargest(5, h)
        final_h = set()
        for elem in h1: # distinct
            final_h.add(elem[1][0])
            final_h.add(elem[1][1])
        final_h = list(final_h)

        actual_labels = set(y_data[i])

        n = 3.0
        heap_err += error(final_h, actual_labels, n)
        pred_err += error(predictions, actual_labels, n)

    with open('./stats/task6_stats.dat', 'a') as f:
        f.write("\nerror type " + test_type)
        f.write("\nerror n " +  str(n))
        f.write('\nwith correlation (heap) ' + str(heap_err/(end-start)))
        f.write("\noriginal pred " + str(pred_err/(end-start)))
        f.write("\n------")
    f.close()
    # end evaluate_model

alpha = 1.0 # for p(a,b) to power alpha
evaluate_model('TRAIN', 0, split_index, alpha)
evaluate_model('TEST', split_index, len(x), alpha)
