import json
import mlpy
import numpy
import math
import itertools
import heapq

FILE_NAME = './stats/task12_twtr_kfold.dat'

max_label = 0
with open(FILE_NAME, 'a') as f:
    f.write('\n\n\n\n=========================================================')
f.close()

# just running for fixed size of 3000 with k-fold of 10%
SIZE = '1000'
x_file_name = './data/task3_social_tfidf2d_list' + SIZE + '.json'
# x_file_name = './data/task2_tfidf2d_list' + SIZE + '.json'
x_file_data = open(x_file_name).read()
x_main = numpy.array(json.loads(x_file_data))

# y_file_name = './data/task2_y' + SIZE + '.json'
y_file_name = './data/task3_social_y' + SIZE + '.json'

corr_err = 0.0
base_err = 0.0

numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json'))) # get number of labels
# generate y as a 2D array where we indicate presence/absence of each label
#y is column major
arr = [-1]*len(x_main)
y_main = numpy.array([list(arr) for _ in xrange(numLabels)])

# corr_y = numpy.load('./stats/correlation_mat.npy') # load correlation matrix
corr_y = numpy.load('./stats/social_correlation_mat.npy') # load social correlation matrix

y_open = open(y_file_name)
y_data = numpy.array(json.loads(y_open.read()))
y_open.close()
for i,label_list in enumerate(y_data):
    for label in label_list:
        y_main[label][i]=1

x_len = len(x_main)
x_len_10 = x_len/10

for k_fold in range(x_len_10, x_len+1, x_len_10):
    SOLVER = 'l2r_l2loss_svc' # best l2r_l2loss_svc
    C = 1 # default is 1
    
    ALPHA = 0.25 # for sqrt(p(a,b)) to power alpha, between 0.3 and 0.4 gives best result
    N = 3 # 1/importance given to overprediction
    K =  'len pred' # 4 # top K pairs from correlated probabilities
    TRAIN_PERCENT = 0.9 # split index

    with open(FILE_NAME, 'a') as f:    
        f.write('\n\n--------')
        f.write('\nK-FOLD ' + str(10))
        f.write('\nK-FOLD iteration' + str(k_fold))
        f.write('\nSOLVER ' + SOLVER)
        f.write('\nSIZE ' + SIZE)
        f.write('\nALPHA without SQRT ' + str(ALPHA))
        f.write('\nERROR-N ' + str(N))
        f.write('\nPAIRS-CHOSEN-K ' + str(K))
        f.write('\nSPLIT-INDEX ' + str(TRAIN_PERCENT))
    f.close()

    stop = k_fold
    start = k_fold - x_len_10
    print len(x_main),len(x_main[0])
    print len(y_main),len(y_main[0])
    # print start, stop, x_len
    # print  y_main[:,stop:] 
    # print y_main[:,stop:]
    # print y_main[:,start:stop]
    y1 = numpy.concatenate((y_main[:,:start], y_main[:,stop:]), axis=1)
    y = numpy.concatenate((y1, y_main[:,start:stop]), axis=1)
    x1 = numpy.concatenate((x_main[:start,:], x_main[stop:,:]), axis=0)
    x = numpy.concatenate((x1, x_main[start:stop,:]), axis=0)
    print len(x), len(x[0])
    print len(y), len(y[0])

    split_index = int(math.floor(TRAIN_PERCENT * len(x)))

    # x = x_main
    # y = y_main

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
            probabilities = []

            for idx in xrange(len(temp_probs)):
                label = temp_probs[idx][2]
                if label[0] == 1:
                    probabilities.append((1 - temp_probs[idx][0], temp_probs[idx][1]))
                else:
                    probabilities.append((temp_probs[idx][0], temp_probs[idx][1]))
            
            predictions = [idx for idx,model in enumerate(models) if model.pred(x[i])==1 ]
            global max_label
            max_label = max(len(predictions), max_label)
            probabilities.sort(reverse=True)
            final_h = list()
            h = [] # heap
            if len(predictions) == 0:
                final_h = probabilities[:1]
            elif len(predictions) == 1:
                final_h = predictions
            else:
                
                probabilities =  probabilities[:10] # consider 10 with highest Pr


                # considering tuples from correlation matrix and choosing top K=4
                
                pairs = itertools.combinations(probabilities,2)
                for a,b in pairs:
                    p_a_b = math.sqrt(corr_y[a[1]][b[1]] * corr_y[b[1]][a[1]] )
                    # p_a_b *= math.sqrt(corr_y_train[a[1]][b[1]] * corr_y_train[b[1]][a[1]] )
                    p_a_b = math.pow(p_a_b, alpha)

                    pr = a[0]*b[0]*p_a_b
                    key = (a[1], b[1])
                    heapq.heappush(h, (pr, key))
                K = len(predictions)
                # k_map = [0,0,1,2,3,4,4,4,4,4,4]
                # K = k_map[len(predictions)]
                h1 = heapq.nlargest(K, h) # how many to choose in the end
                
                for elem in h1: # distinct
                    if not elem[1][0] in final_h:
                        final_h.append(elem[1][0])
                    if not elem[1][1] in final_h:  
                        final_h.append(elem[1][1])
                num_baseline = len(predictions)
                before_final_h = final_h
                final_h = list(before_final_h[:max(num_baseline,1)])
                # final_h = list(set(final_h))
            # end else
            
            top_four_pairs = list()
            h2 = heapq.nlargest(4, h)
            for elem in h2: # distinct
                if not elem[1][0] in top_four_pairs:
                    top_four_pairs.append(elem[1][0])
                if not elem[1][1] in top_four_pairs:  
                    top_four_pairs.append(elem[1][1])

            actual_labels = set(y_data[i])

            global N, corr_err, base_err
            cur_heap_err = error(final_h, actual_labels, N)
            cur_pred_err = error(predictions, actual_labels, N)
            heap_err += cur_heap_err
            pred_err += cur_pred_err


            corr_err += heap_err*(0.1)
            base_err += pred_err*0.1

        with open(FILE_NAME, 'a') as f:
            f.write("\nerror type " + test_type)
            f.write('\nwith correlation (heap) ' + str(heap_err/(end-start)))
            f.write("\noriginal pred " + str(pred_err/(end-start)))
            f.write("\n--------")
        f.close()
        # end evaluate_model

    evaluate_model('TRAIN', 0, split_index, ALPHA)
    evaluate_model('TEST', split_index, len(x), ALPHA)
    break

print "corr", corr_err
print "base", base_err