
test_idx = [[0,24,31,42,59,80,95],[7,25,34,44,60,81,96],[10,26,35,45,67,86,98],[14,29,36,49,70,91,101],[23,30,41,51,78,94]]
import random
print(len(test_idx))
for i in range(len(test_idx)):
    train = []
    test = []
    test1 = []
    test_tmp = []
    for l in range(len(test_idx)):
        if l!=i:
            test_tmp.extend(test_idx[l])
    #print(test_tmp)
    for j in range(103):
        if j not in test_idx[i]:
            train.append(j)
    for k in train:
        if k not in test_tmp:
            test.append(k)
    if i<2:
        test1 = random.sample(test,14)
        #print(test1)
        test_idx[i].extend(test1)
    else:
        test1 = random.sample(test,13)
        #print(test1)
        test_idx[i].extend(test1)
    test_idx[i].sort()
train_idx = []

for i in range(len(test_idx)):
    train=[]
    print(test_idx[i])
    print()
    for j in range(103):
        if j not in test_idx[i]:
            print(j)
            train.append(j)
    print(train)
    train_idx.append(train)
    print(train_idx)
print(train_idx)
print(test_idx)
