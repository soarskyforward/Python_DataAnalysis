# Python_DataAnalysis
 python数据分析

 - [Python_DataAnalysis](#python-dataanalysis)
   * [第一章 准备工作](#--------)
       - [重要的python数据分析库](#---python-----)
       - [安装或升级Python包](#-----python-)
       - [引入惯例](#----)
   * [python语法基础](#python----)
       - [Python概念和语言机制](#python-------)
       - [控制流](#---)
   * [第3章 Python的数据结构、函数和文件](#-3--python-----------)
       - [数据结构和序列](#-------)
       - [生成器](#---)
       - [文件与操作系统](#-------)
   * [第4章 NumPy基础：数组和矢量计算](#-4--numpy----------)
       - [NumPy的ndarray：一种多维数组对象](#numpy-ndarray---------)
       - [NumPy数组的运算](#numpy-----)
       - [布尔型索引](#-----)
       - [利用数组进行数据处理](#----------)
       - [用于数组的文件输入输出](#-----------)
       - [线性代数](#----)
   * [pandas入门](#pandas--)
       - [pandas的数据结构](#pandas-----)
       - [基本功能](#----)
       - [DataFrame和Series之间的运算](#dataframe-series-----)
       - [排序和排名](#-----)
       - [汇总和计算描述统计](#---------)
   * [第6章 数据加载、存储与文件格式](#-6--------------)
       - [读写文本格式的数据](#---------)
   * [第7章 数据清洗和准备](#-7---------)
       - [处理缺失数据](#------)
       - [数据转换](#----)
       - [字符串操作](#-----)
   * [第8章 数据规整：聚合、合并和重塑](#-8---------------)
       - [层次化索引](#-----)
       - [合并数据集](#-----)

 >参考书目《利用Python进行数据分析》

 ## 第一章 准备工作

 #### 重要的python数据分析库

 Numpy
 pandas
 matplotlib
 IPython和Jupyter
 SciPy
 seaborn
 statsmodels

 #### 安装或升级Python包
 ```
 pip install package_name
 pip install --upgrade package_name

 ```
 #### 引入惯例
 ```
 import numpy as np
 import matplotlib.pyplot as plt
 import pandas as pd
 import seaborn as sns
 import statsmodels as sm
 ```

 ## python语法基础

 >从事数据分析和科学计算的人却会使用IPython，一个强化的Python解释器，或Jupyter notebooks，一个网页代码笔记本

 ```
 #运行ipython
 ipython

 #运行jupyter notebook
 jupyter notebook

 #%run命令运行所有的Python程序
 %run hello_world.py

 #集成Matplotlib
 #ipython
 %matplotlib
 #jupyter
 %matplotlib inline
 ```

 #### Python概念和语言机制
 >万物皆对象,Python语言的一个重要特性就是它的对象模型的一致性。每个数字、字符串、数据结构、函数、类、模块等等，都是在Python解释器的自有“盒子”内，它被认为是Python对象。

 ```
 #动态引用，强类型
 a = 5
 type(a)
 a = 'foo'
 type(a)

 #isinstance函数检查对象是某个类型的实例
 isinstance(a, int)
 isinstance(a, (int, float))
 ```
 >属性和方法,Python的对象通常都有属性（其它存储在对象内部的Python对象）和方法（对象的附属函数可以访问对象的内部数据）。可以用obj.attribute_name访问属性和方法


 >鸭子类型,经常地，你可能不关心对象的类型，只关心对象是否有某些方法或用途。这通常被称为“鸭子类型”，来自“走起来像鸭子、叫起来像鸭子，那么它就是鸭子”的说法。

 ```
 def isiterable(obj):
     try:
 	   iter(obj)
 	   return True
     except TypeError:
 	   return False

 isiterable('a string')
 ```

 可变与不可变对象
 >Python中的大多数对象，比如列表、字典、NumPy数组，和用户定义的类型（类），都是可变的。意味着这些对象或包含的值可以被修改：

 ```
 a_list = ['foo', 1, 2]
 a_list[2] = (3, 4)

 #其它的，例如字符串和元组，是不可变的
 a_tuple = (2, 5, (4, 5))
 a_tuple[1] = 'four'
 #返回typeerror
 ```

 数值类型
 >python的主要数值类型为int和float
 ```
 ival = 123
 fval = 3.14
 ```

 字符串

 ```
 a = 'one way of writing a string'
 b = "another way"
 c = """
 This is a longer string that
 spans multiple lines
 """

 #Python的字符串是不可变的，不能修改字符串
 a = 'this is a string'
 b = a.replace('string', 'longer string')
 a
 #变量a并没有被修改

 #许多Python对象使用str函数可以被转化为字符串
 a = 3.2
 str(a)

 ```
 布尔值
 ```
 #python中的布尔值有两个
 True and True
 False or True
 ```

 类型转换
 ```
 #str、bool、int和float也是函数，可以用来转换类型
 s = '3.14159'
 fval = float(s）
 type(fval)
 int(fval)
 bool(fval)
 ```
 日期和时间

 >Python内建的datetime模块提供了datetime、date和time类型。datetime类型结合了date和time，是最常使用的

 ```
 from datetime import datetime, date, time

 dt = datetime(2011, 10, 29, 20, 30, 21)

 dt.day
 dt.year

 dt.date()
 dt.time()

 #strftime方法可以将datetime格式化为字符串
 dt.strftime('%m/%d/%Y %H:%M')

 #strptime可以将字符串转换成datetime对象
 dt.strptime('20091031', '%Y%m%d')
 ```
 #### 控制流
 if,elif和else
 ```
 if x < 0:
     print('It's negative')
 elif x == 0:
     print('Equal to zero')
 elif 0 < x < 5:
     print('Positive but smaller than 5')
 else:
     print('Positive and larger than or equal to 5')
 ```

 for循环
 >for循环是在一个集合（列表或元组）中进行迭代，或者就是一个迭代器

 ```
 #用continue使for循环提前，跳过剩下的部分
 sequence = [1, 2, None, 4, None, 5]
 total = 0
for value in sequence:
  if value is None:
    continue
  total += value


 #用break跳出for循环
 sequence = [1, 2, 0, 4, 6, 5, 2, 1]
 total_until_5 = 0
 for value in sequence:
     if value == 5:
         break
     total_until_5 += value
 ```

 while循环
 ```
 x = 256
 total = 0
 while x > 0:
     if total > 500:
         break
     total += x
     x = x // 2
 ```

 pass
 >pass是Python中的非操作语句。代码块不需要任何动作时可以使用（作为未执行代码的占位符）
 ```
 def function():
    pass
 ```

 range
 ```
 #range的三个参数是（起点，终点，步进）

 list(range(0, 20, 2))
 ```
 三元表达式
 ```
 value = true_expr if condition else false_expr

 if condition:
     value = true_expt
 else:
     value = false_expr
 ```


 ## 第3章 Python的数据结构、函数和文件

 #### 数据结构和序列

 元祖
 >元组是一个固定长度，不可改变的Python序列对象。创建元组的最简单方式，是用逗号分隔一列值

 ```
 tup = 2, 3, 4
 #tuple可以将任意序列或迭代器转换成元组
 tup = tuple('string')

 #拆分元组
 seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
 for a, b, c in seq:
     print('a = {0}, b = {1}, c = {2}'.format(a,b,c))
 ```
 >Python最近新增了更多高级的元组拆分功能，允许从元组的开头“摘取”几个元素。它使用了特殊的语法*rest

 ```
values = 1, 2, 3, 4, 5
a, b ,rest* =  values
 ```
 列表

 >与元组对比，列表的长度可变、内容可以被修改。你可以用方括号定义，或用list函数

 ```
 a_list = [1,2,3,None]
 tup = ('foo', 'bar')
 b_list = list(tup)

 #删除和添加元素

 #append在列表末尾添加
 b_list.append('sdf'）
 #insert可以在特定位置插入
 b_list.insert(1,'red')
 #insert的逆运算是pop，它移除并返回指定位置的元素
 b_list.pop(2)
 #remove去除某个值，remove会先寻找第一个值并除去
 b_list.remove('foo')

 #用in可以检查列表是否包含某个值
 'sdf' in b_list
 'wei' not in b_list

 #串联和组合列表
 [4, None, 'foo'] + [7, 8, (2, 3)]

 x = [4, None, 'foo']
 x.extend([2,3,3])


 #排序
 a = [4,46,2,5]
 a.sort()
 b = ['saw', 'small', 'He', 'foxes', 'six']
 b.sort(key = len)

 #切片
 seq = [7, 2, 3, 7, 5, 6, 0, 1]
 seq[1:5]
 seq[3:4] = [6, 3]
 #负数表明从后向前切片
 seq[-4:]
 seq{-4:-2]
 #在第二个冒号后面使用step，可以隔数取一个元素
 seq[::2]
 #将列表或元祖颠倒过来
 seq[::-1]
 ```

 序列函数

 >Python内建了一个enumerate函数，可以返回(i, value)元组序列
 sorted函数可以从任意序列的元素返回一个新的排好序的列表
 zip可以将多个列表、元组或其它序列成对组合成一个元组列表
 reversed可以从后向前迭代一个序列


 ```
 #enumerate函数
 some_list = ['foo', 'bar', 'baz']
 mapping = {}

 for i, v in enumerate(some_list):
     mapping[v] = i

 #sort函数
 sorted('horse race')

 #zip函数
 seq1 = ['foo', 'bar', 'baz']
 seq2 = ['one', 'two', 'three']
 zipped = zip(seq1, seq2)

 #zip的常见用法之一是同时迭代多个序列，可能结合enumerate使用
 for i, (a,b) in enumerate(zip(seq1, seq2)):
      print('{0}: {1}, {2}'.format(i, a, b))
 #zip还可以用来解压序列
 pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'),('Schilling', 'Curt')]
 first_names, last_names = zip(*pitchers)

 #reversed函数
 list(reversed(range(10)))
 ```

 字典
 >字典可能是Python最为重要的数据结构。它更为常见的名字是哈希映射或关联数组。它是键值对的大小可变集合，键和值都是Python对象。创建字典的方法之一是使用尖括号，用冒号分隔键和值：

 ```
 empty_dict = {}
 d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}

 #可以用del关键字或pop方法（返回值的同时删除键）删除值
 del d1[5]
 ret = d1.pop('a')

 #keys和values是字典的键和值的迭代器方法。
 list(d1.keys())
 list(d1.values())

 #update方法可以将一个字典与另一个融合
 d1.update({'b' : 'foo', 'c' : 12})

 #用序列创建字典
 mapping = {}
 for key, value in zip(key_list, value_list):
     mapping[key] = value

 mapping = dict(zip(range(10), reversed(range(10))))

 #默认值
 if key in some_dict:
     value = some_dict[key]
 else:
     value = default_value

 value = some_dict.get(key, default_value)


  for word in words:
      letter = word[0]
      if letter not in by_letter:
          by_letter[letter] = [word]
      else:
          by_letter[letter].append(word)

 for word in words:
     letter = word[0]
     by_letter.setdefault(letter, []).append(word)
 ```

 集合
 >集合是无序的不可重复的元素的集合。你可以把它当做字典，但是只有键没有值。可以用两种方式创建集合：通过set函数或使用尖括号set语句

 ```
 set([2, 2, 2, 1, 3, 3])
 #集合支持合并、交集、差分和对称差等数学集合运算
 a.union(b) #a | b
 a.intersection(b)	#a & b
 ```

 列表、集合和字典推导式
 >list_comp = [expr for val in collection if condition]
 dict_comp = {key-expr : value-expr for value in collection if condition}
 set_comp = {expr for value in collection if condition}


 ```
 strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
 [x.upper() for x in strings if len(x) > 2]
 ```
 函数
 >函数使用def关键字声明，用return关键字返回值

 ```
 def my_function(x, y, z=1.5):
     if z > 1:
         return z * (x + y)
     else:
         return z / (x + y)
 ```
 命名空间、作用域，和局部函数
 >函数可以访问两种不同作用域中的变量：全局（global）和局部（local）。Python有一种更科学的用于描述变量作用域的名称，即命名空间（namespace）。任何在函数中赋值的变量默认都是被分配到局部命名空间（local namespace）中的。局部命名空间是在函数被调用时创建的，函数参数会立即填入该命名空间。在函数执行完毕之后，局部命名空间就会被销毁



 匿名（lambda）函数
 ```
 def short_function(x):
     return x * 2

 equiv_anon = lambda x: x * 2
 ```

 #### 生成器
 >能以一种一致的方式对序列进行迭代（比如列表中的对象或文件中的行）是Python的一个重要特点。这是通过一种叫做迭代器协议（iterator protocol，它是一种使对象可迭代的通用方式）的方式实现的，一个原生的使对象可迭代的方法。

 ```
 some_dict = {'a': 1, 'b': 2, 'c': 3}
 for key in some_dict:
     print(key)

 dict_itered = iter(some_dict)
 list(dict_itered)
 ```
 >生成器（generator）是构造新的可迭代对象的一种简单方式。一般的函数执行之后只会返回单个值，而生成器则是以延迟的方式返回一个值序列，即每返回一个值之后暂停，直到下一个值被请求时再继续。要创建一个生成器，只需将函数中的return替换为yeild即可

 ```
 def squares(n = 10):
     print('Generating squares from 1 to{0}'.formar(n ** 2))
     for i in range(1,n+1):
 	yield i ** 2

 #生成器表达式
 gen = (x ** 2 for x in range(100))

 ```
 #### 文件与操作系统
 ```
path = 'examples/segismundo.txt'
f = open(path)
 ```
 >默认情况下，文件是以只读模式（'r'）打开的。然后，我们就可以像处理列表那样来处理这个文件句柄f了，比如对行进行迭代
 ```
 for line in f:
 	pass
 ```

 ```
 #使用open创建文件对象，一定要用close关闭它。关闭文件可以返回操作系统资源
 f.close()

 with open(path) as f:
 	lines = [x.rstrip() for x in f]
 ```
 ```
 #向文件写入，可以使用文件的write或writelines方法
 with open('tmp.txt', 'w') as handle:
     handle.writelines(x for x in open(path) if len(x) > 1)

 with open('tmp.txt') as f:
     lines = f.readlines()
 ```

 ## 第4章 NumPy基础：数组和矢量计算

 >NumPy是在一个连续的内存块中存储数据，独立于其他Python内置对象。NumPy的C语言编写的算法库可以操作内存，而不必进行类型检查或其它前期工作。比起Python的内置序列，NumPy数组使用的内存更少。
 NumPy可以在整个数组上执行复杂的计算，而不需要Python的for循环。

 #### NumPy的ndarray：一种多维数组对象
 ```
 import numpy as np

 data = np.random.randn(2,3)
 data + data
 data * 10

 data.shape
 data.dtype

 #创建ndarray
 data1 = [6, 7.5, 8, 0, 1]
 arr1 = np.array(data1)
 data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
 arr2 = np.array(data2)
 arr2.ndim
 ```
 >zeros和ones分别可以创建指定长度或形状的全0或全1数组。empty可以创建一个没有任何具体值的数组

 ```
 np.zeros(10)
 np.zeros((2,4))
 np.empty((2,2,3))
 ```
 ```
 #arange是Python内置函数range的数组版
 np.arange(10)
 ```
 ndarray的数据类型

 ```
 arr1 = np.array([1, 2, 3], dtype=np.float64)
 arr2 = np.array([1, 2, 3], dtype=np.int32)

 #通过ndarray的astype方法明确地将一个数组从一个dtype转换成另一个dtype
 float_arr = arr2.astype(np.float64)
 ```

 #### NumPy数组的运算
 ```
 arr = np.array([[1., 2., 3.], [4., 5., 6.]])
 arr * arr
 arr - arr
 1 / arr

 #基本的索引和切片
 arr = np.arange(10)
 arr[2]
 arr[2:5]
 ```
 >跟列表最重要的区别在于，数组切片是原始数组的视图。这意味着数据不会被复制，视图上的任何修改都会直接反映到源数组上。
 >如果你想要得到的是ndarray切片的一份副本而非视图，就需要明确地进行复制操作，例如arr[5:8].copy()。


 #### 布尔型索引
 ```
 names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
 data = np.random.randn(7,4)

 names == 'Bob'
 data[names == 'Bob']

 #要选择除"Bob"以外的其他值，既可以使用不等于符号（!=），也可以通过~对条件进行否定
 names != 'Bob'
 data[~(name == 'Bob')]

 #通过布尔型数组设置值是一种经常用到的手段
 data[data < 0] = 0
 ```
 数组转置和轴对换
 ```
 arr = np.arange(15).reshape((3,5))
 arr.T

 #在进行矩阵计算时，经常需要用到该操作，比如利用np.dot计算矩阵内积
  np.dot(arr.T, arr)
 ```
 通用函数：快速的元素级数组函数
 >通用函数（即ufunc）是一种对ndarray中的数据执行元素级运算的函数。你可以将其看做简单函数（接受一个或多个标量值，并产生一个或多个标量值）的矢量化包装器。

 ```
 arr = np.arange(10)
 np.sqrt(arr)
 np.exp(arr)

 #这些都是一元（unary）ufunc。另外一些（如add或maximum）接受2个数组（因此也叫二元（binary）ufunc），并返回一个结果数组
 x = np.random.randn(8)
 y = np.random.randn(8)
 np.maximum(x,y)
 ```
 #### 利用数组进行数据处理

 ```
  xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
 yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
 cond = np.array([True, False, True, True, False])
 result = [(x if c else y) for x , y , c in zip(xarr, yarr ,cond)]

 #使用np.where，则可以将该功能写得非常简洁
 result = np.where(cond, xarr, yarr)
 np.where(arr > 0, 2, -2)
 ```

 数学和统计方法
 ```
 arr = np.random.randn(5, 4)
 #arr.mean(1)是“计算行的平均值”，arr.sum(0)是“计算每列的和”
 arr.mean(axis=1)
 arr.sum(axis=0)
 ```
 唯一化以及其它的集合逻辑
 ```
 names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
 np.unique(names)
 ```
 #### 用于数组的文件输入输出
 ```
 arr = np.arange(10)
 np.save('some_array.npy', arr)
 np.load('some_array.npy')
 ```
 #### 线性代数

 ```
 #x.dot(y)等价于np.dot(x, y)
 x = np.array([[1., 2., 3.], [4., 5., 6.]])
 y = np.array([[6., 23.], [-1, 7], [8, 9]])

 x.dot(y)
 np.dot(x,y)
 ```


 ## pandas入门
 >虽然pandas采用了大量的NumPy编码风格，但二者最大的不同是pandas是专门为处理表格和混杂数据设计的。而NumPy更适合处理统一的数值数组数据。
 ```
 import pandas as pd
 from pandas import Series, DataFrame
 ```
 #### pandas的数据结构
 Series
 ```
 obj = pd.Series([4,7,-5,3])
 obj.index
 obj.values

 obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
 obj2['a']
 obj2['d'] = 6

 'b' in obj2
 'e' in obj2

 sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
 obj3 = pd.Series(sdata)

 states = ['California', 'Ohio', 'Oregon', 'Texas']
 obj4 = pd.Series(sdata, index=states)

 #pandas的isnull和notnull函数可用于检测缺失数据：
 pd.isnull(obj4)
 pd.notnull(obj4)
 obj4.isnull()

 #Series对象本身及其索引都有一个name属性，该属性跟pandas其他的关键功能关系非常密切
 obj4.name = 'population'
 obj4.index.name = 'state'
 ```

 DataFrame
 >DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔值等）。DataFrame既有行索引也有列索引，它可以被看做由Series组成的字典（共用同一个索引）。DataFrame中的数据是以一个或多个二维块存放的（而不是列表、字典或别的一维数据结构）。

```
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DateFrame(data)

#对于特别大的DataFrame，head方法会选取前五行
frame.head()

#如果指定了列序列，则DataFrame的列就会按照指定顺序进行排列
pd.DataFrame(data, columns=['year', 'state', 'pop'])

#通过类似字典标记的方式或属性的方式，可以将DataFrame的列获取为一个Series
frame2 = pd.DataFrame(data, columns = ['year', 'state', 'pop', 'debt'],
                      index = ['one', 'two', 'three', 'four','five','six'])
frame2.columns
frame2['state']
frame2.year
```
>frame2[column]适用于任何列的名，但是frame2.column只有在列名是一个合理的Python变量名时才适用。

```
#行也可以通过位置或名称的方式进行获取，比如用loc属性
frame2.loc['three']

frame2['debt'] = 16.5
frame2['debt'] = np.arange(6.)

#关键字del用于删除列
frame2['eastern'] = frame2.state == 'Ohio'
del frame2['eastern']
frame2.columns
```

>通过索引方式返回的列只是相应数据的视图而已，并不是副本。因此，对返回的Series所做的任何就地修改全都会反映到源DataFrame上。通过Series的copy方法即可指定复制列。

```
#如果嵌套字典传给DataFrame，pandas就会被解释为：外层字典的键作为列，内层键则作为行索引
pop = {'Nevada': {2001: 2.4, 2002: 2.9},'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DateFrame(pop)

#使用类似NumPy数组的方法，对DataFrame进行转置
frame3.T

#如果设置了DataFrame的index和columns的name属性，则这些信息也会被显示出来
frame3.index.name = 'year'
frame3.columns.name = 'state'
```

pandas索引对象
>pandas的索引对象负责管理轴标签和其他元数据（比如轴名称等）。构建Series或DataFrame时，所用到的任何数组或其他序列的标签都会被转换成一个Index

```
obj = pd.Series(range(3), index = ['a', 'b', 'c'])
index = obj.index

#index对象不可变
index[1] = 'd' #TypeError

#不可变可以使Index对象在多个数据结构之间安全共享
labels = pd.Index(np.arange(3))
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
```

#### 基本功能
重新索引
```
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
```
>对于时间序列这样的有序数据，重新索引时可能需要做一些插值处理。method选项即可达到此目的，例如，使用ffill可以实现前向值填充
```
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method = 'ffill')
```

丢弃指定轴上的项
```
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
obj.drop(['d', 'c'])

#通过传递axis=1或axis='columns'可以删除列的值
data.drop('two', axis = 1)
data.drop('two', axis = 'columns')
```

索引、选取和过滤
```
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj[['b', 'a', 'd']]
obj[2:4]
obj[obj < 2]

#利用标签的切片运算与普通的Python切片运算不同，其末端是包含的：
obj['b':'c']
```

用loc和iloc进行选取
```
data.loc['Colorado', ['two', 'three']]
data.iloc[2, [3, 0, 1]]
data.loc[:'Utah', 'two']
data.iloc[:, :3][data.three > 5]
```

算术运算和数据对齐
>pandas最重要的一个功能是，它可以对不同索引的对象进行算术运算。在将对象相加时，如果存在不同的索引对，则结果的索引就是该索引对的并集。对于有数据库经验的用户，这就像在索引标签上进行自动外连接。

```
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1 + s2
```
在算术方法中填充值
```
df1.add(df2, fill_value=0)
```


#### DataFrame和Series之间的运算

```
arr = np.arange(12.).reshape((3,4))
arr - arr[0]
```

>当我们从arr减去arr[0]，每一行都会执行这个操作。这就叫做广播（broadcasting）


```
series2 = pd.Series(range(3), index=['b', 'e', 'f'])
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                 columns=list('bde'),
                 index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series3 = frame['d']
frame.sub(series3, axis='index')
```

>传入的轴号就是希望匹配的轴。在本例中，我们的目的是匹配DataFrame的行索引（axis='index' or axis=0）并进行广播。

函数应用和映射
```
frame = pd.DataFrame(np.random.randn(4,3), columns = list('abc'),
                      index = ['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)

f = lambda x: x.max() - x.min()
frame.apply(f)
frame.apply(f, axis = 'columns')
```

#### 排序和排名
```
obj = pd.Series(range(4), index = ['d', 'a', 'b','c'])
obj.sort_index()

frame = pd.DataFrame(np.arange(8).reshape((2,4)),index=['three', 'one'],
                      columns=['d', 'a', 'b', 'c'])
frame.sort_index(axis = 1,ascending=False)

obj = pd.Series([4, 7, -3, 2])
obj.sort_values()

obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()
obj.rank(method = 'first')
```

#### 汇总和计算描述统计
>NA值会自动被排除，除非整个切片（这里指的是行或列）都是NA。通过skipna选项可以禁用该功能

```
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                    [np.nan, np.nan], [0.75, -1.3]],
                   index=['a', 'b', 'c', 'd'],
                   columns=['one', 'two'])
df.sum()
df.mean(axis = 'columns', skipna = False)

df.cumsum() #累加
df.cumprod()  #累乘

df.describe()
```
>Series的corr方法用于计算两个Series中重叠的、非NA的、按索引对齐的值的相关系数。与此类似，cov用于计算协方差

```
import pandas_datareader.data as web
all_data = {ticker: web.get_data_yahoo(ticker)
            for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']}

price = pd.DataFrame({ticker: data['Adj Close']
                     for ticker, data in all_data.items()})
volume = pd.DataFrame({ticker: data['Volume']
                      for ticker, data in all_data.items()})
returns = price.pct_change()

 returns['MSFT'].corr(returns['IBM'])
 returns['MSFT'].cov(returns['IBM'])
```


## 第6章 数据加载、存储与文件格式

#### 读写文本格式的数据
```
df = pd.read_csv('examples/ex1.csv')
df = pd.read_table('examples/ex1.csv', sep = ',')

#并不是所有文件都有标题行,如下例
#你可以让pandas为其分配默认的列名，也可以自己定义列名
pd.read_csv('examples/ex2.csv', header=None)
names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('examples/ex2.csv', names = names)

#假设你希望将message列做成DataFrame的索引。你可以明确表示要将该列放到索引4的位置上，
也可以通过index_col参数指定"message"
pd.read_csv('examples/ex2.csv', names = names, index_col = 'message')
```
>有些情况下，有些表格可能不是用固定的分隔符去分隔字段的（比如空白符或其它模式）,
虽然可以手动对数据进行规整，这里的字段是被数量不同的空白字符间隔开的。这种情况下，
你可以传递一个正则表达式作为read_table的分隔符。可以用正则表达式表达为\s+
```
result = pd.read_table('examples\ex3.txt', seq = '\s+')
```

>这些解析器函数还有许多参数可以帮助你处理各种各样的异形文件格式。比如说，
可以用skiprows跳过文件的第一行、第三行和第四行
```
pd.read_csv('examples/ex4.csv', skiprows = [0,2,3])
```

>缺失值处理是文件解析任务中的一个重要组成部分。缺失数据经常是要么没有（空字符串），
要么用某个标记值表示。默认情况下，pandas会用一组经常出现的标记值进行识别，比如NA及NULL
```
result = pd.read_csv('examples/ex5.csv', na_values=['NULL'])
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv('examples/ex5.csv', na_values=sentinels)
```

逐块读取文本文件
```
pd.options.display.max_rows = 10

result = pd.read_csv('examples/ex6.csv')
#如果只想读取几行（避免读取整个文件），通过nrows进行指定即可
result = pd.read_csv('examples/ex6.csv', nrows = 5)
```
读取Microsoft Excel文件
```
xlsx = pd.ExcelFile('examples/ex1.xlsx')
pd.read_excel(xlsx, 'Sheet1')
```

将数据写出到文本格式
```
data = pd.read_csv('examples/ex5.csv')
data.to_csv('examples/out.csv',index=False, header=False)
data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])
```

## 第7章 数据清洗和准备

#### 处理缺失数据
>在pandas中，我们采用了R语言中的惯用法，即将缺失值表示为NA，它表示不可用not available。

```
string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data.isnull()
```
滤除缺失数据
```
from numpy import nan as NA

data = pd.Series([1, NA, 3.5, NA, 7])
data.dropna()
#等价于
data[data.notnull()]

data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
                      [NA, NA, NA], [NA, 6.5, 3.]])
#dropna默认丢弃任何含有缺失值的行
cleaned = data.dropna()
data.dropna(how = 'all')
data.dropna(axis=1, how='all')                  
```
填充缺失数据
```
df = pd.DateFrame(np.random.randn(7,3))
df.fillna(0)
#通过一个字典调用fillna，就可以实现对不同的列填充不同的值
df.fillna({1: 0.5, 2: 0})
```

#### 数据转换
移除重复数据
```
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
                      'k2': [1, 1, 2, 3, 3, 4, 4]})
#duplicated函数返回一个布尔型Series，表示各行是否是重复行（前面出现过的行）
data.duplicated()
data.drop_duplicates()
#也可以指定部分列进行重复项判断
data.drop_duplicates(['k1'])
```
利用函数或映射进行数据转换
```
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                            'Pastrami', 'corned beef', 'Bacon',
                            'pastrami', 'honey ham', 'nova lox'],
                      'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
meat_to_animal = {'bacon': 'pig','pulled pork': 'pig','pastrami': 'cow','corned beef': 'cow','honey ham': 'pig','nova lox': 'salmon'}

lowercased = data['food'].str.lower()
data['animal'] = lowercased.map(meat_to_animal)
```
替换值
```
#-999这个值可能是一个表示缺失数据的标记值。要将其替换为pandas能够理解的NA值，我们可以利用replace来产生一个新的Series（除非传入inplace=True）
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data.replace(-999, np.nan)
data.replace([-999, -1000], [np.nan, 0])
```
重命名轴索引
```
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                     index=['Ohio', 'Colorado', 'New York'],
                     columns=['one', 'two', 'three', 'four'])
transform = lambda x: x[:4].upper()
data.index.map(transfrom)

data.rename(index = str.title, columns = str.upper)
```

离散化和面元划分
>为了便于分析，连续数据常常被离散化或拆分为“面元”（bin）。

```
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)

cats.codes
cats.categories
pd.value_counts(cats)

#通过传递一个列表或数组到labels，设置自己的面元名称
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels = group_names)
```
>qcut是一个非常类似于cut的函数，它可以根据样本分位数对数据进行面元划分。根据数据的分布情况，cut可能无法使各个面元中含有相同数量的数据点。而qcut由于使用的是样本分位数，因此可以得到大小基本相等的面元

```
data = np.random.randn(1000)
cats = pd.qcut(data, 4)
pd.value_counts(cats)
#与cut类似，你也可以传递自定义的分位数
 pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
```

检测和过滤异常值
```
data = pd.DateFrame(np.random.randn(1000,4))
data.describe()
col = data[2]
col[np.abs(col) > 3]
#np.sign(data)可以生成1和-1
data[np.abs(data) > 3] = np.sign(data) * 3
```
#### 字符串操作
```
val = 'a,b,  guido'
val.split(',')
pieces = [x.strip() for x in val.split(',')]
first, second, third = pieces
':'.join(pieces)
val.replace(',', '::')
```

## 第8章 数据规整：聚合、合并和重塑
#### 层次化索引
>层次化索引（hierarchical indexing）是pandas的一项重要功能，它使你能在一个轴上拥有多个（两个以上）索引级别。抽象点说，它使你能以低维度形式处理高维度数据。

```
data = pd.Series(np.random.randn(9),
                  index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                        [1, 2, 3, 1, 3, 1, 2, 2, 3]])

data.index
data['b']
data['b':'c']
data.loc[['b','d']]

#还可以在“内层”中进行选取
data.loc[:, 2]
```
>层次化索引在数据重塑和基于分组的操作（如透视表生成）中扮演着重要的角色。例如，可以通过unstack方法将这段数据重新安排到一个DataFrame中

```
data.unstack()
#unstack的逆运算是stack
data.instack().stack()
```

重拍与分级排序
>有时，你需要重新调整某条轴上各级别的顺序，或根据指定级别上的值对数据进行排序。swaplevel接受两个级别编号或名称，并返回一个互换了级别的新对象（但数据不会发生变化）

```
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                    index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                     columns=[['Ohio', 'Ohio', 'Colorado'],
                               ['Green', 'Red', 'Green']])
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']

frame.swaplevel('key1','key2')
frame.sort_index(level=1)
frame.swaplevel(0, 1).sort_index(level=0)

#根据级别汇总统计
frame.sum(level = 'key2')
frame.sum(level = 'color', axis = 1)
```

使用DataFrame的列进行索引
```
frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                       'c': ['one', 'one', 'one', 'two', 'two',
                             'two', 'two'],
                       'd': [0, 1, 2, 0, 1, 2, 3]})
#DataFrame的set_index函数会将其一个或多个列转换为行索引
frame2 = frame.set_index(['c', 'd'])
#默认情况下，那些列会从DataFrame中移除，但也可以将其保留下来
frame.set_index(['c', 'd'], drop = False)

#reset_index的功能跟set_index刚好相反，层次化索引的级别会被转移到列里面
frame2.reset_index()
```

#### 合并数据集
- pandas.merge可根据一个或多个键将不同DataFrame中的行连接起来。SQL或其他关系型数据库的用户对此应该会比较熟悉，因为它实现的就是数据库的join操作。
- pandas.concat可以沿着一条轴将多个对象堆叠到一起。
- 实例方法combine_first可以将重复数据拼接在一起，用一个对象中的值填充另一个对象中的缺失值。

```
#数据库风格的DataFrame合并
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})

df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                     'data2': range(3)})
pd.merge(df1, df2)
pd.merge(df1, df2, on = 'key')                

#如果两个对象的列名不同，也可以分别进行指定

```
