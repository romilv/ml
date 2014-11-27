import json
import mlpy
import numpy
import math
import itertools
import heapq


# AMAZON TO AMAZON
SOLVER = 'l2r_l2loss_svc' # best l2r_l2loss_svc
C = 1 # default is 1
SIZE = '3000'
ALPHA = 0.1 # for p(a,b) to power alpha
N = 3 # 1/importance given to overprediction
K = 4 # top K pairs from correlated probabilities

with open('./stats/task6_stats.dat', 'a') as f:
    f.write('\n\n--------')
    f.write('\nSOLVER ' + SOLVER)
    f.write('\nC ' + str(C))
    f.write('\nSIZE ' + SIZE)
    f.write('\nALPHA ' + str(ALPHA))
    f.write('\nERROR-N ' + str(N))
    f.write('\nPAIRS-CHOSEN-K ' + str(K))
    f.write('\nSMOOTHING 1/100 of max') # 1 or 1/100 of max
    # f.write('\nSMOOTHING ' + str(K)) # 1 or 1/100 of max
f.close()

x_file_name = './data/task2_tfidf2d_list' + SIZE + '.json'
x_file_data = open(x_file_name).read()
x = numpy.array(json.loads(x_file_data))

y_file_name = './data/task2_y' + SIZE + '.json'
#Get number of labels
numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json')))
#Generate y as a 2D array where we indicate presence/absence of each label
#y is column major
#Also build the correlation matrix
arr = [-1]*len(x)
y = numpy.array([list(arr) for _ in xrange(numLabels)])
corr_y = numpy.array([list([0.0]*numLabels) for _ in xrange(numLabels)])

# total_count = 0.0

y_open = open(y_file_name)
y_data = numpy.array(json.loads(y_open.read()))
for i,label_list in enumerate(y_data):
    for label in label_list:
        y[label][i]=1
        # total_count += 1.0
    for item in itertools.permutations(label_list,2):
        corr_y[item[0]][item[1]]+=1.0

# average_label_count = total_count/len(x)

numpy.savetxt('./data/task6_corr_orig.dat', corr_y, fmt='%.4d', newline='\n\n')

# Normalize the correlation matrix
for idx,label in enumerate(corr_y):
    smoothing_factor = max(label)*0.01 # 1 or (1/100) of max in row
    label += smoothing_factor
    total = sum(label)
    if total==0:
        continue
    corr_y[idx]*=1.0/total

numpy.savetxt('./data/task6_corr_norm.dat', corr_y, fmt='%.6f', newline='\n\n')

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
        temp_probs = [(1-model.pred_probability(x[i])[0], idx, model.labels()) for idx,model in enumerate(models)] 
        # tprobabilities = [(model.pred_probability(x[i]), idx, model.labels()) for idx,model in enumerate(models)]
        probabilities = []

        for idx in xrange(len(temp_probs)):
            label = temp_probs[idx][2]
            if label[0] == 1:
                probabilities.append((1 - temp_probs[idx][0], temp_probs[idx][1]))
            else:
                probabilities.append((temp_probs[idx][0], temp_probs[idx][1]))
        
        # predictions2 = [(idx, x[i]) for idx,model in enumerate(models) if model.pred(x[i])==1 ]
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
        h1 = heapq.nlargest(K, h) # how many to choose in the end
        final_h = set()
        for elem in h1: # distinct
            final_h.add(elem[1][0])
            final_h.add(elem[1][1])
        final_h = list(final_h)

        actual_labels = set(y_data[i])

        cur_heap_err = error(final_h, actual_labels, N)
        cur_pred_err = error(predictions, actual_labels, N)
        heap_err += cur_heap_err
        pred_err += cur_pred_err

        if test_type == 'TEST':
            if cur_heap_err > cur_pred_err:
                with open('./data/task6_heap_debug.dat', 'a') as f:
                    f.write("\n\n------------------")
                    f.write('\nheap err ' + str(cur_heap_err))
                    f.write('\npred err ' + str(cur_pred_err))
                    f.write('\nheap ' + str(final_h))
                    f.write('\npred ' + str(predictions))
                    f.write('\nactual ' + str(actual_labels))
                    f.write('\nprobs ' + str(probabilities))
                    # f.write('\ninput ' + str(temp_probs))
                    f.write("\n------------------")
                f.close()
            else:
                with open('./data/task6_heap_correct.dat', 'a') as f:
                    f.write("\n\n------------------")
                    f.write('\nheap err ' + str(cur_heap_err))
                    f.write('\npred err ' + str(cur_pred_err))
                    f.write('\nheap ' + str(final_h))
                    f.write('\npred ' + str(predictions))
                    f.write('\nactual ' + str(actual_labels))
                    f.write('\nprobs ' + str(probabilities))
                    f.write("\n------------------")
                f.close()


    with open('./stats/task6_stats.dat', 'a') as f:
        f.write("\nerror type " + test_type)
        f.write('\nwith correlation (heap) ' + str(heap_err/(end-start)))
        f.write("\noriginal pred " + str(pred_err/(end-start)))
        f.write("\n------")
    f.close()
    # end evaluate_model


with open('./data/task6_heap_debug.dat', 'w') as f:
    pass
f.close()
with open('./data/task6_heap_correct.dat', 'w') as f:
    pass
f.close()

evaluate_model('TRAIN', 0, split_index, ALPHA)
evaluate_model('TEST', split_index, len(x), ALPHA)
