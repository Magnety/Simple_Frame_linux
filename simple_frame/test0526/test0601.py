test= [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]]
print(test[0:2])
for i in range(12):
    if i not in test[0]:
        print(i)