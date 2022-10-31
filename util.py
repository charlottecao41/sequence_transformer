from __future__ import print_function
import sys
import copy
import random
import numpy as np
from collections import Counter
from collections import defaultdict

def data_partition(fname,context_length):
    usernum = 0
    itemnum = 0
    count_test = 0
    break_u=-1
    User = defaultdict(list)
    u_i_label = defaultdict()
    user_train = {}
    user_valid = {}
    user_test = {}
    feedback_num_list=[]
    # assume user/item index starting from 1
    f = open(fname, 'r')
    for line in f:
        u, i, label = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        label=int(label)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        if break_u == u:
            continue

        if u not in user_train:
            user_train[u]=[]
        if u not in user_valid:
            user_valid[u]=[]
        if u not in user_test:
            user_test[u]=[]
        if u <= 10006:
            user_train[u].append(i)
        else:
            if label !=context_length:
                user_train[u].append(i)
            else:
                user_test[u].append(i)
                count_test = count_test + 1
                feedback_num_list.append(len(user_train[u]))
                break_u=u

    print(Counter(feedback_num_list))
    print(np.mean(feedback_num_list))
    print("count_test:"+str(count_test))
    return [user_train, user_valid, user_test, usernum, itemnum]

