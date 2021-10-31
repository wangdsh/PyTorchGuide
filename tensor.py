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

# ------------------------------
#    张量与数组的转换
# ------------------------------

# ------------------------------
#    数学和比较操作
# ------------------------------

# ------------------------------
#    索引
# ------------------------------

# ------------------------------
#    改变张量形状
# ------------------------------

