import csv
import math

TRAIN_FILE='./car/train.csv'
TEST_FILE='./car/test.csv'


ATTRIBUTES={'buying':['vhigh', 'high', 'med', 'low'],\
    'maint':['vhigh', 'high', 'med', 'low'],\
        'doors':['2', '3', '4', '5more'],\
            'persons':['2', '4', 'more'],\
                'lug_boot':['small', 'med', 'big'],\
                    'safety':['low', 'med', 'high']}
columns=['buying','maint','doors','persons','lug_boot','safety','label']
attributes=['buying','maint','doors','persons','lug_boot','safety']


def def_data(labels):
    dic={}
    for label in labels:
        if label not in dic.keys():
            dic[label]={}
    return dic

def init_data(reader):
    global columns
    dataset=[]
    for sample in reader:
        atrs={}
        for i in range(len(columns)):
            if columns[i] not in atrs.keys():
                atrs[columns[i]]=sample[i]
        dataset.append(atrs)
    return dataset

def get_subset(S,attr,value):
    dataset=[]
    for sample in S:
        if sample[attr]==value:
            res={}
            for label in sample.keys():
                if label==attr:
                    continue
                res[label]=sample[label]
            dataset.append(res)
    return dataset

def Log(a):
    return -a*math.log2(a)      
        
def Entropy(dic,sum):
    res=0
    for key in dic.keys():
        if key=='sum':
            continue
        res+=Log(dic[key]/sum)
    return res

def ME(dic,sum):
    return sorted(dic.items(), key=lambda x: x[1])[0][1]/sum

def GI(dic,sum):
    res=1
    for key in dic.keys():
        if key=='sum':
            continue
        res-=(dic[key]/sum)**2
    return res

def Gain(entropy,attribute,data,algo='Entropy'):
    gain=entropy
    for value in data[attribute].keys():
        if value=='sum':
            continue
        if algo=='Entropy':
            nbr=Entropy(data[attribute][value],data[attribute][value]['sum'])
        elif algo=='ME':
            nbr=ME(data[attribute][value],data[attribute][value]['sum'])
        elif algo=='GI':
            nbr=GI(data[attribute][value],data[attribute][value]['sum'])
        gain-=(data[attribute][value]['sum']/data['sum'])*nbr
    return gain

class Node:
    def __init__(self,label) -> None:
        self.data=label
        self.children={}

def analyse(S):
    global columns
    col=[]
    for label in S[0].keys():
        col.append(label)
    attr=def_data(col)
    attr['sum']=0

    for sample in S:
        attr['sum']+=1
        for label in col:
            if label==columns[-1]:
                if sample[label] not in attr[label].keys():
                    attr[label][sample[label]]=0
                attr[label][sample[label]]+=1
            else:
                if sample[label] not in attr[label].keys():
                    attr[label][sample[label]]={'sum':0}
                attr[label][sample[label]]['sum']+=1
                if sample[columns[-1]] not in attr[label][sample[label]].keys():
                    attr[label][sample[label]][sample[columns[-1]]]=0
                attr[label][sample[label]][sample[columns[-1]]]+=1
    return attr
    
def get_best_attr(attr,algo='Entropy'):
    col=[]
    for label in attr.keys():
        if label!='sum':
            col.append(label)
    result={}
    for a in col:
        if a=='label':
            continue
        if algo=='Entropy':
            result[a]=Gain(Entropy(attr['label'],attr['sum']),a,attr,algo='Entropy')
        elif algo=='ME':
            result[a]=Gain(ME(attr['label'],attr['sum']),a,attr,algo='ME')
        elif algo=='GI':
            result[a]=Gain(GI(attr['label'],attr['sum']),a,attr,algo='GI')
    return sorted(result.items(), key=lambda x: x[1], reverse=True)[0][0]

def ID3(S,Attributes,layer,max_layer=0,algo='Entropy'):
    label_dic={}
    for sample in S:
        if sample['label'] not in label_dic.keys():
            label_dic[sample['label']]=0
        label_dic[sample['label']]+=1
    label_sorted=sorted(label_dic.items(), key=lambda x: x[1], reverse=True)
    if len(label_dic.keys())==1:
        return Node(label_sorted[0][0])
    if len(Attributes)==0:
        return Node(label_sorted[0][0])
    if max_layer>0 and layer>max_layer:
        return Node(label_sorted[0][0])
    
    A=get_best_attr(analyse(S),algo)

    n=Node(A)
    for value in ATTRIBUTES[A]:
        Sv=get_subset(S,A,value)
        #print(Sv)
        if len(Sv)==0:
            n.children[value]=Node(label_sorted[0][0])
        else:
            Attr=[]
            for a in Attributes:
                if a!=A:
                    Attr.append(a)
            n.children[value]= ID3(Sv,Attr,layer+1,max_layer)

    return n


def predict(root,sample):
    if len(root.children)==0:
        return root.data
    else:
        return predict(root.children[sample[root.data]],sample)

def test(root, S):
    corr=0
    summ=0
    for sample in S:
        summ+=1
        y=predict(root,sample)

        if y==sample[columns[-1]]:
            corr+=1
    print('acc: '+str(corr/summ))
    print('err: '+str((summ-corr)/summ))


def run_test(train_data,test_data):
    for i in range(1,7):
        print('layer='+str(i))
        print('Entropy:')
        test(ID3(train_data,attributes,1,i,'Entropy'),test_data)
        print('ME:')
        test(ID3(train_data,attributes,1,i,'ME'),test_data)
        print('GI:')
        test(ID3(train_data,attributes,1,i,'GI'),test_data)

def main():
    train_data_reader=csv.reader(open(TRAIN_FILE,'r'))
    test_data_reader=csv.reader(open(TEST_FILE,'r'))

    train_data=init_data(train_data_reader)
    test_data=init_data(test_data_reader)


    print('run test on testing data:')
    run_test(train_data,test_data)
    print('run test on training data:')
    run_test(train_data,train_data)

if __name__ == '__main__':
    main()