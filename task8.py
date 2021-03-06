import json
import mlpy
import numpy
import math
import itertools
import heapq

STATS_FILE = './stats/task9_corr_stats.dat'
NUM_LABELS = 31

with open(STATS_FILE, 'a') as f:
    f.write('\n\n\n\n--------------------------------')
f.close()

for size_iter in range(1000, 1001, 500):
    SIZE = str(size_iter)

    # AMAZON TO AMAZON
    SOLVER = 'l2r_l2loss_svc' # best l2r_l2loss_svc
    SOLVER_MC = 'mcsvm_cs' # Multi-class SVM
    C = 1 # default is 1
    # SIZE = '3000'
    ALPHA = 0.45 # for sqrt(p(a,b)) to power alpha, between 0.3 and 0.4 gives best result
    N = 3 # 1/importance given to overprediction
    K = 5 # top K pairs from correlated probabilities
    TRAIN_PERCENT = 0.7 # split index

    with open(STATS_FILE, 'a') as f:
        f.write('\n\n--------')
        # f.write('\nSOLVER ' + SOLVER)
        # f.write('\nC ' + str(C))
        f.write('\nSIZE ' + SIZE)
        # f.write('\nALPHA without SQRT ' + str(ALPHA))
        # f.write('\nERROR-N ' + str(N))
        # f.write('\nPAIRS-CHOSEN-K ' + str(K))
        # f.write('\nSPLIT-INDEX ' + str(TRAIN_PERCENT))
    f.close()

    # x_file_name = './data/task3_social_tfidf2d_list' + SIZE + '.json' # twitter
    x_file_name = './data/task2_tfidf2d_list' + SIZE + '.json' # amazon
    x_file_data = open(x_file_name).read()
    x = numpy.array(json.loads(x_file_data))

    y_file_name = './data/task2_y' + SIZE + '.json' # amazon
    # y_file_name = './data/task3_social_y' + SIZE + '.json' # twitter

    numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json'))) # get number of labels
    # generate y as a 2D array where we indicate presence/absence of each label
    #y is column major
    arr = [-1]*len(x)
    y = numpy.array([list(arr) for _ in xrange(numLabels)])

    # corr_y = numpy.load('./stats/correlation_mat.npy') # amazon
    # corr_y = numpy.load('./stats/social_correlation_mat.npy') # twitter

    y_open = open(y_file_name)
    y_data = numpy.array(json.loads(y_open.read()))
    y_open.close()
    for i,label_list in enumerate(y_data):
        for label in label_list:
            y[label][i]=1

    split_index = int(math.floor(TRAIN_PERCENT * len(x)))

    #Train model for all labels
    models = []
    for label in y:
        temp = mlpy.LibLinear(solver_type=SOLVER, C=C)
        temp.learn(x[:split_index],label[:split_index])
        models.append(temp)
        pass

    corr_models = [] # 31x31
    print len(y)
    for iter_i in range(len(y)):
        tem = list()
        for iter_j in range(len(y)):
            if iter_i>=iter_j:
                tem.append(0)
                continue
            tem_labels = y[iter_i] * y[iter_j]
            mdl = mlpy.LibLinear(solver_type=SOLVER, C=C)
            mdl.learn(x[:split_index],tem_labels[:split_index])
            tem.append(mdl)
        corr_models.append(tem)



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
            #Individual probabilities
            temp_probs = [(1-model.pred_probability(x[i])[0], idx, model.labels()) for idx,model in enumerate(models)] 
            probabilities = []

            for idx in xrange(len(temp_probs)):
                label = temp_probs[idx][2]
                if label[0] == 1:
                    probabilities.append((1 - temp_probs[idx][0], temp_probs[idx][1]))
                else:
                    probabilities.append((temp_probs[idx][0], temp_probs[idx][1]))
            
            predictions = [idx for idx,model in enumerate(models) if model.pred(x[i])==1 ]
            probabilities.sort(reverse=True)
            probabilities =  probabilities[:10] # consider 10 with highest Pr


            # considering tuples from correlation matrix and choosing top K=4
            h = [] # heap
            pairs = itertools.combinations(probabilities,2)
            for a,b in pairs:
                # print a,' ',b
                corr_mdl = corr_models[min(a[1],b[1])][max(a[1],b[1])]
                lOrder = corr_mdl.labels()
                probs = corr_mdl.pred_probability(x[i])
                p_a_b= probs[0] if lOrder[0]>lOrder[1] else 1-probs[0]
                # p_a_b = math.sqrt(corr_y[a[1]][b[1]] * corr_y[b[1]][a[1]])
                p_a_b = math.pow(p_a_b, alpha)
                p_ab = math.pow(a[0]*b[0],1-alpha)
                pr = p_ab*p_a_b
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

            # if test_type == 'TEST':
            #     if cur_heap_err > cur_pred_err:
            #         with open('./data/task6_heap_debug.dat', 'a') as f:
            #             f.write("\n\n------------------")
            #             f.write('\nheap err ' + str(cur_heap_err))
            #             f.write('\npred err ' + str(cur_pred_err))
            #             f.write('\nheap ' + str(final_h))
            #             f.write('\npred ' + str(predictions))
            #             f.write('\nactual ' + str(actual_labels))
            #             f.write('\nprobs ' + str(probabilities))
            #             # f.write('\ninput ' + str(temp_probs))
            #             f.write("\n------------------")
            #         f.close()
            #     else:
            #         with open('./data/task6_heap_correct.dat', 'a') as f:
            #             f.write("\n\n------------------")
            #             f.write('\nheap err ' + str(cur_heap_err))
            #             f.write('\npred err ' + str(cur_pred_err))
            #             f.write('\nheap ' + str(final_h))
            #             f.write('\npred ' + str(predictions))
            #             f.write('\nactual ' + str(actual_labels))
            #             f.write('\nprobs ' + str(probabilities))
            #             f.write("\n------------------")
            #         f.close()

        with open(STATS_FILE, 'a') as f:
            f.write("\nerror type " + test_type)
            f.write('\nwith correlation (heap) ' + str(heap_err/(end-start)))
            f.write("\noriginal pred " + str(pred_err/(end-start)))
            f.write("\n--------")
        f.close()
        # end evaluate_model


    # with open('./data/task6_heap_debug.dat', 'w') as f:
    #     pass
    # f.close()
    # with open('./data/task6_heap_correct.dat', 'w') as f:
    #     pass
    # f.close()

    evaluate_model('TRAIN', 0, split_index, ALPHA)
    evaluate_model('TEST', split_index, len(x), ALPHA)