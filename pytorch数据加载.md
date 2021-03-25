# pytorch数据加载

## 1使用数据加载器的目的

在深度学习中，数据量通常很大，不可能一次性在模型中进行计算与反向传播，所以需要对数据进行随机的打乱，把数据处理成一个个的batch

## 2数据集类：

`torch.utils.data.Dataset`

`__len__`方法，能够实现全局的len（）方法获取其中的元素个数
`__getitem__`方法，能够通过传入索引的放视获取数据，例如通过dataset[i]获取其中的第I条数据

```python
import torch 
from torch.utils.data import Dataset

data_path = 'C:/Users/11024/Desktop/torch/zhaopin.txt'

#数据集类
class ACEDataset(Dataset):
    def __init__(self):
        self.lines = open(data_path).readlines()

    def __getitem__(self, index):
        #获取索引对应位置的一条数据
        return self.lines[index]

    def __len__(self): 
        #返回数据的总数量
        return len(self.lines)


if __name__ == '__main__':
    m_dataset = ACEDataset()
    print(m_dataset[150])
```

之后对dataset进行实例化，可以迭代获取其中的数据

```python
for i in range(len(m_dataset)):
        print(i,m_dataset[i])
```



## 3.迭代数据集

使用上述方法可以进行数据的读取，但是许多内容还没有实现

- 批处理数据（batching the data）
- 打乱数据（shuffle）
- 使用多线程（muti-processing）并行加载数据

在pytorch中`torch.utils.data.Dataloader`提供了上述的方法

`Dataloader`使用方法：

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset=m_dataset,batch_size=10,shuffle=True,num_workers=2,drop_last=True)
    for index,(label,content) in enumerate(dataloader):
        print(index,label,content)
        print("*"*100)
```

参数：

1. `dataset`：提前定义的dataset实例
2. `batch_size`:一次传入的batch大小
3. `shuffle`：是否打乱数据
4. `num_workers`:加载数据的线程数

结果：

```
****************************************************************************************************
181 ('高级数据分析', '数据分析师,', '数据分析决策', '数据分析（H', '数据分析（实', '高级数据分析', '数据分析专家', '海外酒店数据', '商业数据分析', '数据分析师,') ('师,10k-20k,3-5年,武汉,福韵数据服务有限公司,150-500人,洪山区,A轮,"移动互联
网,数据服务",数据分析,1,0', '13k-20k,1-3年,北京,北京链家,2000人以上,朝阳区,不需要融资,房产家居,其他数据分析,1,25', '岗 
（风险管理部）,15k-25k,3-5年,北京,阳光产险信保事业部,2000人以上,朝阳区,不需要融资,金融,数据分析,1,100', 'RBI）,16k-30k,5-10年,北京,美团点评,2000人以上,朝阳区,上市公司,消费生活,BI,0,0', '习／校招）,4k-8k,应届毕业生,北京,数美,150-500人,朝阳
区,B轮,"企业服务,数据服务",数据分析,1,100', '师,12k-20k,5-10年,武汉,优品财富管理有限公司,500-2000人,江夏区,未融资,金融,数据分析,2,5', ',20k-35k,5-10年,北京,360,2000人以上,朝阳区,上市公司,信息安全,数据分析,2,75', '分析师（收益定价方向）,10k-20k,1-3年,上海,携程,2000人以上,长宁区,上市公司,旅游,数据分析,0,0', ',6k-8k,3-5年,杭州,盛锋咨询,50-150人,拱墅区,未融资
,数据服务,数据分析,0,0', '15k-20k,1-3年,北京,美团点评,2000人以上,朝阳区,上市公司,消费生活,数据分析,3,100')
****************************************************************************************************
182 ('数据分析师,', '数据分析师,', '高级数据分析', '数据分析师,', '数据分析,1', '数据分析师,', '数据分析师,', '数据分析师（', '数据分析师,', '数据分析师,') ('10k-15k,3-5年,深圳,华策,50-150人,福田区,未融资,移动互联网,数据分析,0,0', '10k-20k,1-3年,上海,DataStory,150-500人,杨浦区,B轮,数据服务,数据分析,2,100', '师,20k-30k,3-5年,杭州,嘿豆商城,150-500人,下沙,未
融资,"移动互联网,电商",数据分析,0,0', '15k-20k,3-5年,北京,彩讯股份,500-2000人,东城区,上市公司,"移动互联网,消费生活",数 
据分析,0,0', '0k-15k,1-3年,深圳,汇合,2000人以上,罗湖区,不需要融资,"企业服务,金融",数据分析,1,63', '13k-25k,1-3年,上海, 
萨摩耶金服,150-500人,浦东新区,C轮,金融,数据分析,0,0', '20k-35k,3-5年,上海,星合金融科技,50-150人,黄浦区,不需要融资,"移动
互联网,金融",数据分析,0,0', '数据策略方向）,15k-30k,3-5年,北京,车好多集团,2000人以上,朝阳区,D轮及以上,消费生活,数据分析
,1,100', '8k-12k,1-3年,上海,眸事网,150-500人,杨浦区,天使轮,"移动互联网,企业服务",数据分析,1,100', '7k-13k,1-3年,上海,通
联数据,150-500人,虹口区,不需要融资,金融,数据分析,0,0')
****************************************************************************************************

```

 注意：

```
len(dataset) = 数据集的样本数

len(dataloader) = math.ceil(样本数/batch_size)向上取整
```

## 4. Pytorch自带的数据集

pytorch中自带的数据集由两个上层`api`提供，`torchvision`和`torchtext`

其中

1. `torchvision`提供了对图片数据处理相关的api和数据

   数据位置：`torchvision.datasets`,如`torchvision.datasets.MNIST`(手写数字识别)

2. `torchtext`提供了文本数据处理的api和数据

   数据位置：`torchtext.datasets`,如`torchtext.datasets.IMDB`(电影评论数据集)

我们以MNIST为例子，准备好`Datset`实例然后交给`Dataloader`，组成batch

<img src="C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210323211032073.png" alt="image-20210323211032073" style="zoom:80%;" />

## 5.手写数字识别

### 1 .准备`MNIST`数据集的dataset和`dataloader`

准备训练集

```python
dataset = torchvision.datasets.MNIST('./data',train=True,
   				                     download=True,transforms=torch.transforms.Compose([torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,),(0.3081,))
                        ]))

train_dataloader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True)
```

准备测试集：

```python
dataset = torchvision.datasets.MNIST('./data',train=False,
   				                     download=True,transforms=torch.transforms.Compose([torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,),(0.3081,))
                        ]))

train_dataloader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True)
```

### 2.构建模型

补充全连接层：当前一层的神经元和前一层的神经元相互连接，核心操作是
$$
y=wx
$$
模型的构建使用了一个四层的神经网络，其中包括两个全连接层和一个输出层，第一个全连接层会经过激活层的处理，结果交到下一个全连接层，进行变换后输出结果。

注意：

1. 激活函数如何使用
2. 每一层数据的形状
3. 模型的损失函数

#### 2.1 激活函数的使用

常用的激活函数是`Relu`函数，由`import torch.nn.funvtional as F`提供，`F.relu(x)`即可对x进行处理

```python
b = torch.tensor([-2,-1,0,1,2])
F.relu(b)
tensor([0, 0, 0, 1, 2])
```



#### 2.2 模型中的数据形状

1. 原始数据形状[batch_size,1,28,28]
2. 进行形状的修改[batch_size,28*28]，注意batch_size不能改变
3. 第一个全连接层输出[batch_size,28]
4. 激活函数不会修改数据形状
5. 第二个全连接层输出形状[batch_size,10],因为手写数字有10个类别

代码如下

```python
import torch
from torch import nn
import torch.nn.Functional as F

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet,self).__init__()
        self.fc1 = nn.Linear(28*28*1,28)
        self.fc2 = nn.Linear(28,10)
        
    def forward(self,x):
        x = x.view(-1,28*28*1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
```

#### 2.3损失函数

![image-20210325163449490](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325163449490.png)

![image-20210325163635330](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325163635330.png)

在pytorch中有两种方法实现交叉熵损失

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(input,target)
```

```python
output = F.log_softmax(x,dim=1)
loss = criterion(input,target)
```

#### 2.4模型训练

训练流程：

1. 实例化模型，设置模型为训练模式
2. 实例化优化器类，实例化损失函数
3. 获取，遍历`dataloader`
4. 梯度置为0
5. 前向计算
6. 计算损失
7. 反向传播
8. 更新参数

代码：

```python
mnist_net = MnistNet()
optimizer = optim.Adam(mnist_net.parameters(),lr=0.001)
def train(epoch):
    mode = True
    mnist_net.train(mode=mode)
    
    train_dataloader = get_dataloader(train=mode)
    for idx,(data,target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = mnist_net(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if idx % 10==0:
            print('Train Epoch: {} [{}/{} ({:.6f}%)]\tLoss{:.6f}'.format(epoch,idx*len(data),len(train_dataloader.dataset),
                       100.*idx / len(train_dataloader),loss.item()))
```

#### 2.5模型的保存与加载

1. ##### 模型的保存：

   ```python
   torch.save(mnist_net.state_dict(),"model/mnist_net.pt")#保存模型参数
   torch.save(optimizer.state_dict(),"model/mnist_optimizer.pt")#保存优化器参数
   ```

   

2. ##### 模型的加载：

   ```
   mnist_net.load_state_dict(torch.load("model/mnist_net.pt"))
   optimizer.load_state_dict(torch.load("model/mnist_optimizer.pt"))
   ```

   

#### 2.6模型的评估

评估过程与训练过程相似，但是：

1. 不需要计算梯度
2. 需要收集损失和精准率，用来计算平均损失和平均准确率
3. 损失的计算和训练时损失的计算方法相同
4. 精准率的计算：
   - 模型的输出为[batch_size,10]
   - 其中最大值的位置就是其预测值的目标值（预测值进行过`softmax`后变为概率，`softmax`分母都是相同的，分子越大概率就越大）
   - 最大值的获取方法可以使用`torch.max`，返回最大值和最大值的位置、
   - 返回最大值之后，和真实值（[batch_size])进行对比，相同就预测成功

```python
def test():
	test_loss = 0
    correct = 0
    mnist_net.eval()#设置为评估模式
    test_dataloader = get_dataloader(train= False)#获取测试数据集
    with torch.no_grad():#不计算梯度
        for data, taget in enumerate(test_dataloader):
            output= mnist_net(data)
            test_loss +=F.nll_loss(output,target,reduction='sum').item()
            pred = output.data.max(1,keepdim=True)[1]#获取最大值位置[batch_size,1]
            
            correct += pred.eq(target.data.view_as(pred).sum)#预测准备样本数累加
            
    test_loss /= len(test_dataloader.dataset)#计算平均损失
     print('\n Test set: Avg.loss: {:.4f},Accuracy:{}/{} ({:.2f}%)]\n'.format(test_loss,correct,len(test_dataloader.dataset),
                       100.*correct / len(test_dataloader.dataset)))
```

完整代码：

```python
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
#准备数据集
BATCH_SIZE = 64
TEST_BATCH_SIZE = 128

def get_dataloader(train=True):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,),std=(0.3081,))
    ])

    dataset = MNIST('./data',train=True,
                            download=True,transform=transform_fn)

    data_loader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
    return data_loader

#2.构建模型
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet,self).__init__()
        self.fc1 = nn.Linear(28*28*1,28)
        self.fc2 = nn.Linear(28,10)
        
    def forward(self,input):
        x = input.view(-1,28*28*1)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)

        return F.log_softmax(out,dim=-1)

#模型的加载
mnist_net = MnistNet()
optimizer = Adam(mnist_net.parameters(),lr=0.001)
if os.path.exists('./model/model.pkl'):
    mnist_net.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

def train(epoch):
    #实现训练过程
    mode = True
    mnist_net.train(mode=mode)
    
    train_dataloader = get_dataloader(train=mode)
    for idx,(data,target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = mnist_net(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if idx % 10==0:
            #模型的保存
            torch.save(mnist_net.state_dict(),'./model/model.pkl')
            torch.save(optimizer.state_dict(),'./model/optimizer.pkl')

            print('Train Epoch: {} [{}/{} ({:.6f}%)]\tLoss:  {}'.format(epoch,idx*len(data),len(train_dataloader.dataset),
                       100.*idx / len(train_dataloader),loss.item()))


def test():
    test_loss = 0
    correct = 0
    mnist_net.eval()
    test_dataloader = get_dataloader(train= False)#获取测试数据集
    with torch.no_grad():#不计算梯度
        for data, target in test_dataloader:
            output= mnist_net(data)
            test_loss +=F.nll_loss(output,target,reduction='sum').item()
            pred = output.data.max(1,keepdim=True)[1]#获取最大值位置[batch_size,1]
            
            correct += pred.eq(target.data.view_as(pred)).sum()#预测准备样本数累加
            
    test_loss /= len(test_dataloader.dataset)#计算平均损失
    print('\n Test set: Avg.loss: {:.4f},Accuracy:{}/{} ({:.2f}%)]\n'.format(test_loss,correct,len(test_dataloader.dataset),
                       100.*correct / len(test_dataloader.dataset)))


if __name__ == '__main__':
    test()
    for i in range(1):
        train(i)
        test()
```

## 6.循环神经网络和自然语言处理介绍

目标：

1. 知道`token`和`tokenization`
2. 知道N-gram的概念和作用
3. 知道文本向量化表示的方法

### 1.文本的`tokenization`

#### 1.1概念和工具介绍

`tokenizaton`就是分词，分出的每一个词称为token

常见的分词工具：`jieba分词`，清华大学分词工具`thulac`

#### 1.2中英文分词方法

- 句子转化为词语

  我爱深度学习 ------>[我，爱，深度学习]

- 句子转化单个字

  我爱深度学习 ------>[我，爱，深，度，学，习]

### 2. N-gram表示方法

句子可以用单个字或词表示，但是有时候可以用2个或者3个词来表示

N-gram一组一组的单词，其中N表示一起使用的词的数量

例如：

```python
text = '深度学习（英语：deep learning)是机器学习的分支,是一种以人工神经网络为架构对数据进行表征学习的算法'
[cuted[i:i+2] for i in range(len(cuted)-1)] #N-gram N=2
```

```
cuted
[['深度', '学习'],
 ['学习', '（'],
 ['（', '英语'],
 ['英语', '：'],
 ['：', 'deep'],
 ['deep', ' '],
 [' ', 'learning'],
 ['learning', ')'],
 [')', '是'],
 ['是', '机器'],
 ['机器', '学习'],
 ['学习', '的'],
 ['的', '分支'],
 ['分支', ','],
 [',', '是'],
 ['是', '一种'],
 ['一种', '以'],
 ['以', '人工神经网络'],
 ['人工神经网络', '为'],
 ['为', '架构'],
 ['架构', '对'],
 ['对', '数据'],
 ['数据', '进行'],
 ['进行', '表征'],
 ['表征', '学习'],
 ['学习', '的'],
 ['的', '算法']]
 '一种',
 '以',
 '人工神经网络',
 '为',
 '架构',
 '对',
 '数据',
 '进行',
 '表征',
 '学习',
 '的',
 '算法']

```

传统机器学习中N-gram效果很好，但是深度学习中如RNN中会自带N-gram效果

### 3.向量化

文本不能直接计算，所以需要将其转化为向量

两种方法：

1. 转为one-hot(效果不好),使用稀疏向量表示文本，占用空间太多。
2. 转为word embedding

#### 3.1 word embedding

浮点型稠密矩阵来表示token，根据词典大小不同向量使用不同的维度，如100，256，300等，其中向量中的每一个值是一个超参数，初始值是随机生成的，之后会在训练过程中进行学习获得

假设文本中有20000个词语，使用one-hot就会有20000*20000的矩阵，大多数位置都是0，但是用word embedding来表示的话，只需要20000*维度，比如20000*300。

如：

| `token` | `num` | `vector`                            |
| ------- | ----- | ----------------------------------- |
| 词1     | 0     | `[w11,w12,w13....w1n]`其中n表示维度 |
| 词2     | 1     | `[w21,w22,w23....w2n]`              |
| 词3     | 2     | `[w31,w32,w33....w3n]`              |
| ...     | ...   | ...                                 |
| 词m     | m     | `[wm1,wm2,wm3....wmn]`              |

我们会把所有文本转化为向量，把句子用向量表示

**但是在中间，会先把token转为数字表示（ID),再把数字用向量表示。**

token----->ID---------->vector

#### 3.2 word embedding API

![image-20210324145854523](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210324145854523.png)

`torch.nn.Embedding(num_embeddings,embedding_dim)`

参数：

- num_embeddings：词典大小
- embedding_dim：每个词用多少维度表示

使用方法：

```python
embedding = nn.Embedding(vocab_size,300)#实例化
input_embeded = embedding(input)
```

#### 3.3数据的形状变化

每个batch中每个句子10个词语 ，经过形状为[20，4]的word embedding后形状为什么？

每个词语用长度为4的向量表示，所以最终句子会变成[batch_size,10,4]的形状，增加的一个维度就是embedding_dim



## 文本情感分类

目标

1. 知道文本处理的基本方法
2. 能够使用数据实现情感分类

## 1.代码

```python
import torch
from torch.utils.data import DataLoader,Dataset
import os 
import re
data_base_path = "IMDB/aclImdb"

def tokenize(content):
    content= re.sub("<.*?>"," ",content)
    filters = ['\.','\t','\n','\x97','\x96','#','$','%','&']
    content = re.sub('|'.join(filters),'',content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens

class ImdbDataset(Dataset):
    def __init__(self,mode):
        super(ImdbDataset,self).__init__()
        if mode == "train":
            text_path = [os.path.join(data_base_path,i) for i in ['train/neg','train/pos']]
        else:
            text_path = [os.path.join(data_base_path,i) for i in ['test/neg','test/pos']]
        self.total_file_path_list = []#所有评论文件的path
        #把所有文件名放入列表
        for i in text_path:
            self.total_file_path_list.extend([os.path.join(i,j) for j in os.listdir(i)])


    def __getitem__(self, index):
        file_path = self.total_file_path_list[index]
        #获取label
        cur_filename = os.path.basename(file_path)
        label = int(cur_filename.split('_')[-1].split(".")[0]) - 1
        text = tokenize(open(file_path).read().strip())
        
        return label, text

    def __len__(self):
        return len(self.total_file_path_list)




def collate_fn(batch):
    batch = list(zip(*batch))
    labels = torch.tensor(batch[0],dtype=torch.int32)
    texts = batch[1]
    del batch
    return labels,texts

dataset = ImdbDataset(mode='train')
dataloader = DataLoader(dataset=dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)
for idx,(label,text) in enumerate (dataloader):
        print(idx)
        print(label)
        print(text)
        break
```

### 3.4文本序列化

在embedding中文本先转成数字再转化为向量，过程如何实现？

考虑把文本中的**每个词和其对应的数字用字典保存，同时实现方法把句子通过字典映射成包含数字的列表**

在实现文本序列化之前，考虑以下几点：

1. 如何使用字典把词语与数字对应
2. 不同的词语出现的次数不尽相同，是否需要对高频或者低频词语进行过滤，以及总的词语数量是否需要限制
3. 得到词典之后如何把句子转化为数字序列，如何把数字序列转化为句子
4. 不同句子长度不同，每个batch的句子应该如何构造成相同的长度（可以对句子进行填充，填充特殊字符）
5. 对于新出现的词语在词典中没有出现怎么办（用特殊字符代替)

思路分析：

1. 对所有句子进行分词
2. 词语存入字典，根据次数对词语过滤，统计次数
3. 实现文本转数字序列的方法
4. 实现数字序列转文本方法 



代码：

```python
import torch
import numpy as np

class word2sequence():

    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    
    UNK = 0
    PAD = 1 

    def __init__(self):
        self.dict = {
            self.UNK_TAG : self.UNK,
            self.PAD_TAG : self.PAD
        }
        self.count = {}

    def fit(self,sentence):
        """把单个句子保存到dict中
            sentence[word1,word2,word3,...]
        """
        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1


    def build_vocab(self,min=5,max=None,max_features=None):
        """
        生成词典
        min：最小出现的次数
        max：最大出现的次数
        max_features:一共保留多少个词语
        return
        """
        #删除count中词频小于min的词
        if min is not None:
            self.count = {word:value for word,value in self.count.items() if value >min  }

        #删除词频大于max的词
        if max is not None:
           self.count = {word:value for word,value in self.count.items() if value <max }
        
        #限制保留的词语数
        if max_features is not None:
            temp=sorted(self.count.items(),key = lambda x : x[-1],reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)#新加入词的ID 等于原来的词的个数加1

        #键和值反转 ID2WORD 
        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len=None):
        """
        句子转化为序列
        sentence[word1,word2,....]
        """
        assert self.fit
        if max_len is not None:
            r = [self.PAD] *max_len
        else:
            r = [self.PAD] * len(sentence)
        if max_len is not None and len(sentence) > max_len:
            sentence = sentence[:max_len]
        return [self.dict.get(word,self.UNK)for word in sentence]
        """for index,word in enumerate(sentence):
            r[index] = self.to_index(word)"""

        return np.array(r,dtype=np.int64)

    def inverse_transform(self,sentence):
        return[self.inverse_dict.get(idx) for idx in indices ]


if __name__=='__main__':
    ws = word2sequence()
    ws.fit(['who','i','am'])
    ws.fit(['who','are','you'])
    ws.build_vocab(min=0)
    print(ws.dict)
    
    ret = ws.transform(['who','the','hell','you','are'],max_len=10)
    print(ret)
```

### 3.5构造模型

代码：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import ws,max_len
from torch.optim import Adam
import torch.nn.functional as F
from model import MyModel
from dataset import dataloader

class MyModel(nn.Module):
    
    def __init__(self):
        super(MyModel,self).__init__()
        self.embedding = nn.Embedding(len(ws),100)
        self.fc1 = nn.Linear(max_len*100,2)

    def forward(self,input):
        """
        输入[batch_size,max_len]
        
        """
        x = self.embedding(input)#经过embedding后[batch_size,max_len,100]
        x = x.view([-1,max_len*100])#batch_size不用管
        output = self.fc1(x)
        return F.log_softmax(output,dim=-1)
    
    
    model = MyModel()
optimizer = Adam(model.parameters(),0.001)

def train(epoch):
    for idx,(input,target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        print(loss.item())

if __name__ == "__main__":
    for i in range(1):
        train(i)
```

## 循环随机网络

### 目标：

1. 循环神经网络 的概念作用
2. 循环神经网络的类型和应用场景
3. LSTM作用和原理
4. GRU作用和原理

![image-20210325152939192](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325152939192.png)

![image-20210325153655644](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325153655644.png)

![image-20210325153747765](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325153747765.png)

![image-20210325153837367](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325153837367.png)



## 2.LSTM

![image-20210325153932859](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325153932859.png)



### 2.1LSTM核心

![image-20210325154030783](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325154030783.png)

### 2.2理解LSTM

#### 2.2.1 遗忘门

![image-20210325154154502](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325154154502.png)



#### 2.2.2输入门

![image-20210325154305760](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325154305760.png)

#### 2.2.3 输出门

![image-20210325154607066](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325154607066.png)



## 2.3.GRU

![image-20210325154712722](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325154712722.png)

### 2.4 双向LSTM

![image-20210325154734969](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325154734969.png)



# 1. `Pytorch`中`LSTM`和`GRU`的模块使用

## 1.1 LSTM

![image-20210325154927666](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325154927666.png)

LSTM和GRU API：

1. `torch.nn`提供
2. `[batch_size,seq_len]`------------->`[batch_size,seq_len,300]`，300就是input_size
3. 默认情况下数据维度为`[seq_len,batch_size,dim]`，batch_size=True时为`[batch_size,seq_len,dim]`

### 1.2 LSTM使用实例：

假设数据输入为[10,20],假设embedding的形状为[100,30]

代码：

```python
batch_size = 10
seq_len = 20
embedding_dim  = 30
word_vocab = 100
hidden_size = 18
num_layer = 2

#准备输入数据
input = torch.randint(low=2,high=100,size=(batch_size,seq_len))
#准备embedding
embedding = torch.nn.Embedding(word_vocab,embedding_dim)
lstm = torch.nn.LSTM(embedding_dim,hidden_Size,num_layer)

#进行embedding操作
embed = embedding(input)
embed = embed.permute(1,0,2)#将embed的维度调换位置（1，0，2）就是batch_size = False

#初始化状态，如果不初始化则torch默认值全部为0
h_0 = torch.rand(num_layer,batch_size,hidden_size)
c_0 = torch.rand(num_layer,batch_size,hidden_size)
output,(h_1,c_1) = lstm(embed,(h_0,c_0))
#output [20,10,18]
#h_1 [2,10,18]
#c_1 [2,10,18]
```



```python
import torch
import torch.nn as nn

batch_size = 10
seq_len = 20
embedding_dim  = 30
word_vocab = 100
hidden_size = 18
num_layer = 1


class My_LSTM(nn.Module):

    def __init__(self):
        super(My_LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers=num_layer,batch_first=True)
        self.embedding = nn.Embedding(word_vocab,embedding_dim)
    
    def forward(self,input):
        input_embeded = self.embedding(input)
        output,(h_n,c_n) = self.lstm(input_embeded)
        return output,(h_n,c_n)



if __name__=='__main__':
    model = My_LSTM()
    input = torch.randint(low=2,high=100,size=(batch_size,seq_len))
    out,(h_2,c_2) = model(input)
    print(out.size())
    print('*'*100)
    print(h_2.size())
    print('*'*100)
    print(c_2.size())
    
    
"""
torch.Size([10, 20, 18])
****************************************************************************************************
torch.Size([1, 10, 18])
****************************************************************************************************
torch.Size([1, 10, 18])"""
```

### 1.3双向LSTM

如果需要使用双向LSTM，只需要把bidriectional设置为True，同时h_0,c_0使用num_layer*2

代码：

```python
import torch
import torch.nn as nn

batch_size = 10
seq_len = 20
embedding_dim  = 30
word_vocab = 100
hidden_size = 18
num_layer = 1


class My_LSTM(nn.Module):

    def __init__(self):
        super(My_LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers=num_layer,batch_first=True,bidirectional=True)
        self.embedding = nn.Embedding(word_vocab,embedding_dim)
    
    def forward(self,input):
        input_embeded = self.embedding(input)
        output,(h_n,c_n) = self.lstm(input_embeded)
        return output,(h_n,c_n)



if __name__=='__main__':
    model = My_LSTM()
    input = torch.randint(low=2,high=100,size=(batch_size,seq_len))
    out,(h_2,c_2) = model(input)
    print(out.size())
    print('*'*100)
    print(h_2.size())
    print('*'*100)
    print(c_2.size())
    
"""输出
torch.Size([10, 20, 36])
****************************************************************************************************
torch.Size([2, 10, 18])
****************************************************************************************************
torch.Size([2, 10, 18])
"""
```

### 1.4 LSTM 与GRU的注意点

![image-20210325164559719](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325164559719.png)

### 1.5 使用LSTM来改进文本分类

代码：

```python
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.modules import dropout
#from dataset import ws
from lib import ws,max_len


class MyModel(nn.Module):
    
    def __init__(self):
        super(MyModel,self).__init__()
        self.hidden_size = 128
        self.embedding_dim = 100
        self.num_layer = 2
        self.bidirectional = True
        self.bi_num = 2 if self.bidirectional else 1
        self.dropout = 0.5
        #以上为超参数

        self.embedding = nn.Embedding(len(ws),self.embedding_dim)#[N,300]
        self.lstm = nn.LSTM(input_size=100,hidden_size=self.hidden_size,num_layers=self.num_layer,batch_first=True,bidirectional=True,dropout=self.dropout)
        self.fc1 = nn.Linear(self.hidden_size*2,2)
        #self.fc2 = nn.Linear(20,2)


    def forward(self,input):
        """
        输入[batch_size,max_len][128,20]
        
        """
        x = self.embedding(input)#经过embedding后[128,20,100]
        #x:[batch_size,max_len,hidden_size*2],h_n[2*2,batch_Size,hidden_size],c_n[2*2,batch_Size,hidden_size]
        x,(h_n,c_n) =self.lstm(x)#[128,20,256]
        #双向LSTM 获取两个方向最后一次的output，进行concatenate

        output_fw = h_n[-2,:,:]#[20,128]
        output_bw = h_n[-1,:,:]#[20,128]
        output = torch.cat([output_fw,output_bw],dim=-1)#[20,256]

        #x =x.view(128,-1)#[128,5120]
        out = self.fc1(output)#[20,2]
        #out = self.fc2(out)
        #out =self.fc2(out)
        return F.log_softmax(out,dim=-1)
```

## Pytorch中的序列化容器

### 目标

1. 知道梯度消失和梯度爆炸的原理和解决办法
2. 能够使用nn.Sequential完成模型的搭建
3. 知道nn.BatchNormoal的使用方法
4. 知道nn.dropout的使用方法

## 1.梯度消失和梯度爆炸

#### 1.1梯度消失

梯度太小，无法进行参数的更新，梯度小到数据类型无法表示出现nan

![image-20210325194225126](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325194225126.png)

![image-20210325194427083](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325194427083.png)

### 1.2梯度爆炸

梯度爆炸：梯度太大，大到数据类型无法显示，出现nan

![image-20210325194501914](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325194501914.png)

### 1.3解决梯度消失或者梯度爆炸的经验

![image-20210325194606298](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325194606298.png)

### 2.`nn.Seauential`

`nn.Seauential`是一个有序的容器，其中传入的是构造器类（用来处理各种input的类），最终input会被sequential中的构造器依次执行

```python
layer = nn.Sequential(
    	nn.Linear(input_dim,n_hidden_l),
    	nn.Relu(True),
    	nn.Linear(n_hidden_l,n_hidden_2),
    	nn.Relu(True),
    	nn.Linear(n_hidden_2,output_dim),
)
```

![image-20210325195209309](C:\Users\11024\AppData\Roaming\Typora\typora-user-images\image-20210325195209309.png)

### 3.`nn.BatchNormld`

规范化，在每个batch训练中，对参数进行归一化的处理，从而达到训练加速的效果。

以sigmoid激活函数为例子，在反向传播过程中，当值为0，1的时候梯度接近0，导致参数更新的幅度很小。但是对数据进行归一化处理后，就会尽可能的把数据拉到[0-1]的范围，从而让参数更新的幅度变大，提高训练速度。

BatchNorm一般放在激活函数之后，既对输入进行激活处理之后再进行batchNorm

```python
layer = nn.Sequential(
    	nn.Linear(input_dim,n_hidden_l),
    
    	nn.Relu(True),
    	nn.BatchNormld(n_hidden_l),
    
    	nn.Linear(n_hidden_l,n_hidden_2),
    	nn.Relu(True),
    	nn.BatchNormld(n_hidden_2),
    
    	nn.Linear(n_hidden_2,output_dim),
)
```

### 4.`nn.Dropout`

增加模型稳定性

解决过拟合问题

可以理解为训练后的模型是多个模型的组合之后的结果，类似随机森林

```python
layer = nn.Sequential(
    	nn.Linear(input_dim,n_hidden_l),
    
    	nn.Relu(True),
    	nn.BatchNormld(n_hidden_l),
    	nn.Dropput(0.3),
    	
    	nn.Linear(n_hidden_l,n_hidden_2),
    	nn.Relu(True),
    	nn.BatchNormld(n_hidden_2),
    	nn.Dropput(0.3),
    
    	nn.Linear(n_hidden_2,output_dim),
)
```

