import numpy as np 


x = np.arange(1024).reshape(2,512)

x_split = []
for i in range(x.shape[0]):
    split1 = np.split(x[i], 8) #8个一组
    a = [] 
    for j in range(8):
        s = np.split(split1[j], 8)#取8组
        a.append(s)
    x_split.append(a)
    #print(len(a))
print(len(x_split),len(x_split[0]),len(x_split[1]))