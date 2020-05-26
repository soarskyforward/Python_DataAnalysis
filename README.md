# Python_DataAnalysis
 python数据分析

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
 >[expr for val in collection if condition]
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
 ```
