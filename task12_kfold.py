import json
import mlpy
import numpy
import math
import itertools
import heapq



max_label = 0

DOMAIN_IS_AMAZON = True # True = Amazon, False = Twitter

# just running for fixed size of 3000 with k-fold of 10%
SIZE = '2000'
FILE_NAME = ''
x_file_name = ''
y_file_name = ''
corr_y = None
ALPHA = 0.0
K = 0
J = 0

if DOMAIN_IS_AMAZON:
    FILE_NAME = './stats/task12_amzn_kfold.dat'
    x_file_name = './data/task2_tfidf2d_list' + SIZE + '.json'
    y_file_name = './data/task2_y' + SIZE + '.json'
    corr_y = numpy.load('./stats/correlation_mat.npy')
    ALPHA = 0.3
    K = 6
    J = 12

else:    
    FILE_NAME = './stats/task12_twtr_kfold.dat'
    x_file_name = './data/task3_social_tfidf2d_list' + SIZE + '.json'
    y_file_name = './data/task3_social_y' + SIZE + '.json'
    corr_y = numpy.load('./stats/social_correlation_mat.npy')
    ALPHA = 0.25
    K = None
    J = 10
    # end else

x_file_data = open(x_file_name).read()
x_main = numpy.array(json.loads(x_file_data))

numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json'))) # get number of labels
# generate y as a 2D array where we indicate presence/absence of each label
#y is column major
arr = [-1]*len(x_main)
y_main = numpy.array([list(arr) for _ in xrange(numLabels)])

y_open = open(y_file_name)
y_data_main = numpy.array(json.loads(y_open.read()))
# y_data accessed in the evaluate_model() method
y_open.close()
for i,label_list in enumerate(y_data_main):
    for label in label_list:
        y_main[label][i]=1


corr_err = 0.0
base_err = 0.0

x_len = len(x_main)
x_len_10 = x_len/10

for k_fold in range(x_len_10, x_len+1, x_len_10):
    SOLVER = 'l2r_l2loss_svc' # best l2r_l2loss_svc
    C = 1 # default is 1
    
    N = 3 # 1/importance given to overprediction

    TRAIN_PERCENT = 0.9 # split index

    stop_index = k_fold
    start_index = k_fold - x_len_10

    # the problem is not with this
    y, x, y_data = None, None, None

    y1 = numpy.concatenate((y_main[:,:start_index], y_main[:,stop_index:]), axis=1)
    y = numpy.concatenate((y1, y_main[:,start_index:stop_index]), axis=1)

    x1 = numpy.concatenate((x_main[:start_index,:], x_main[stop_index:,:]), axis=0)
    x = numpy.concatenate((x1, x_main[start_index:stop_index,:]), axis=0)

    y_data1 = numpy.concatenate((y_data_main[:start_index],y_data_main[stop_index:]), axis=1)
    y_data = numpy.concatenate((y_data1, y_data_main[start_index:stop_index]), axis=1)

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
                probabilities =  probabilities[:J] # consider J highest Pr
                
                pairs = itertools.combinations(probabilities,2)
                for a,b in pairs:
                    p_a_b = math.sqrt(corr_y[a[1]][b[1]] * corr_y[b[1]][a[1]] )
                    # p_a_b *= math.sqrt(corr_y_train[a[1]][b[1]] * corr_y_train[b[1]][a[1]] )
                    p_a_b = math.pow(p_a_b, alpha)

                    pr = a[0]*b[0]*p_a_b
                    key = (a[1], b[1])
                    heapq.heappush(h, (pr, key))

                global K
                if not DOMAIN_IS_AMAZON:
                    K = len(predictions)
                    # end else
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

            actual_labels = set(y_data[i])

            global N, corr_err, base_err
            cur_heap_err = error(final_h, actual_labels, N)
            cur_pred_err = error(predictions, actual_labels, N)
            heap_err += cur_heap_err
            pred_err += cur_pred_err
            # end for loop
        global TRAIN_PERCENT
        corr_err += (heap_err/(end-start))*(1-TRAIN_PERCENT)
        base_err += (pred_err/(end-start))*(1-TRAIN_PERCENT)

    evaluate_model('TRAIN', 0, split_index, ALPHA)
    evaluate_model('TEST', split_index, len(x), ALPHA)

with open(FILE_NAME, 'a') as f:
    f.write('\n\n================================')
    f.write('\nK ' + str(K))
    f.write('\nJ ' + str(J))
    f.write('\nALPHA' + str(ALPHA))
    f.write('\nsize   ' + SIZE)
    f.write("\ncorr error   " + str(corr_err))
    f.write("\nbase   " + str(base_err))
f.close()