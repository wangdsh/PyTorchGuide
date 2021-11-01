import torch

'''
目录
* 初始化张量
* 数据类型转换
* 张量与数组的转换
* 数学和比较操作
* 索引
* 改变张量形状
'''

# ------------------------------
#    初始化张量
# ------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

t = torch.tensor([[1,2,3], [4,5,6]], device=device, requires_grad=False)
# print(t.shape, t.size(), t.dtype, t.device, t.requires_grad)

t = torch.empty(3, 3) # or torch.empty((3, 3))
x = torch.empty_like(t)
t = torch.zeros(3, 5)
t = torch.ones(3, 5)
t = torch.rand(3)
t = torch.eye(3)
t = torch.arange(1, 2.5, 0.5) # tensor([1.0000, 1.5000, 2.0000])
t = torch.linspace(start=-10, end=10, steps=5) # 平均分，张量大小为steps
t = torch.empty((1, 5)).normal_(mean=0, std=1) # 正态分布
t = torch.empty((1, 5)).uniform_() # 均匀分布
t = torch.diag(torch.ones(3)) # 对角矩阵

# ------------------------------
#    数据类型转换
# ------------------------------
# 类型表详见https://pytorch.org/docs/stable/tensors.html
x = torch.arange(4)
# type(x) <class 'torch.Tensor'>
# x.type() torch.LongTensor
# x.dtype torch.int64
t = x.bool()  # torch.bool
t = x.short() # torch.int16
t = x.int()   # torch.int32
t = x.long()  # torch.int64 !!!
t = x.float() # torch.float32 !!!
t = x.double() # torch.float64

# ------------------------------
#    张量与数组的转换
# ------------------------------
import numpy as np
# 数组 -> 张量
t = torch.from_numpy(np.array([1,2,3]))
# 张量 -> 数组
arr = t.numpy()

# ------------------------------
#    数学和比较操作
# ------------------------------
a = torch.tensor([1,2,3])
b = torch.tensor([6,5,4], dtype=torch.float)
# 加法
c1 = torch.add(a, b)
c2 =  a + b
# 减法
c1 = torch.sub(a, b)
c2 = a - b
# 除法
c1 = torch.div(a, b)
c2 = a / b

# 指数操作
c1 = a.pow(2)
c2 = a ** 2

# 比较
c = a > 0
c = a < 0
c = a <= 0

# 就地操作
c = torch.zeros(3)
c.add_(a) # tensor([1., 2., 3.])
c += a    # tensor([2., 4., 6.])
c2 = c + a # tensor([3., 6., 9.]) # 非就地操作

# ------------------------------
#    乘法(因重要而单列)
# ------------------------------
# 矩阵乘法
m1 = torch.randn((2, 3))
m2 = torch.randn((3, 4))
m = torch.mm(m1, m2) # torch.Size([2, 4])

# 元素级乘
c = a * b

# 点乘(乘积之和)
c = torch.dot(a.float(), b)

# 矩阵指数
c = torch.randn(3, 3).matrix_power(3)

# 批矩阵乘法
c = torch.bmm(torch.randn(32, 3, 4), torch.randn(32, 4, 5)) # [32, 3, 5]

# 广播
c = torch.rand(5, 5) - torch.rand(1, 5)

# ------------------------------
#    索引
# ------------------------------
batch_size, features = 32, 28
t = torch.rand(batch_size, features)
x = t[0] # t[0, :]
x = t[1, 0:10]
t[0, 0] = 6

rows = [1, 3]
cols = [2, 5]
x = t[rows, cols]
x = t[torch.tensor(rows), torch.tensor(cols)]

t = torch.arange(10)
indices = [1, 5, 7]
x = t[indices]

x = t[(t<3) | (t>=8)]
x = t[t.remainder(2) == 0]

x = torch.where(t<5, t, t*2)
x = torch.tensor([1, 2, 2, 3, 3]).unique()

x = t.dim()
x = t.numel()

# ------------------------------
#    改变张量形状
# ------------------------------

# ------------------------------
#    其它操作
# ------------------------------
t = torch.rand(5, 5)

x = torch.sum(t, dim=0, keepdim=False) # torch.Size([5])
values, indices = torch.max(t, dim=1) # torch.min
x = torch.abs(torch.tensor([-1, 2, -3]))
x = torch.argmax(t, dim=1) # torch.argmin
x = torch.eq(torch.tensor([1, 3]), torch.tensor([2, 3]))
x = torch.mean(t, dim=1)
values, indices = torch.sort(t, dim=1, descending=False)
x = torch.clamp(t, min=0) # 夹紧
x = torch.any(t.bool(), dim=1)
x = torch.all(t.bool(), dim=1)

