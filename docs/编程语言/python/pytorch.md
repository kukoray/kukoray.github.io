# pytorch常用手册



## 随机数种子设置

```python
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)   #设置random库
    np.random.seed(seed) # 设置numpy库
    torch.manual_seed(seed) # 设置torch库——CPU
    torch.cuda.manual_seed(seed) # 设置torch库——GPU
    torch.cuda.manual_seed_all(seed) # 设置torch库——GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```





## 数据读取

<img src="https://s2.loli.net/2023/07/10/ksBKjP9DbH7oA6Q.png" alt="image-20230710234942705" style="zoom: 67%;" />

### Dataset







### Dataloader







判断模型梯度

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.linear1 = nn.Linear(2, 1)
    def forward(self, x):
        # 定义模型的前向传播逻辑
        x = self.linear(x)
        return x
model = SimpleModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 输入数据
inputs = torch.tensor([[1.0, 2.0]])

# 前向传播
outputs = model(inputs)

# 计算损失
loss = criterion(outputs, torch.tensor([[0.0]]))

# 反向传播和参数更新
loss.backward()
optimizer.step()

# 输出参数的属性
for param in model.parameters():
    print(f'Parameter: {param}')
    print(f' - Data: {param.data}')
    print(f' - Gradient: {param.grad}')
    print(f' - Requires Gradient: {param.requires_grad}')

```





在PyTorch中，`model.state_dict()`方法返回的是模型中所有可学习参数（需要梯度的参数）的当前状态，即参数的数值，而不包括参数的梯度信息。这是为了方便保存和加载模型的参数状态而设计的。模型的梯度信息在反向传播过程中被计算，但在`state_dict()`中是不包含的。







## BN层

![关于pytorch中BN层（具体实现）的一些小细节](https://pica.zhimg.com/70/v2-870657a8bb025c4d0f2c04d453aa73e1_1440w.image?source=172ae18b&biz_tag=Post)



BN层的输出Y与输入X之间的关系是：Y = (X - running_mean) / sqrt(running_var + eps) * gamma + beta，此不赘言。其中gamma、beta为可学习参数（在pytorch中分别改叫weight和bias），训练时通过反向传播更新；

而running_mean、running_var则是在前向时先由X计算出mean和var，再由mean和var以动量momentum来更新running_mean和running_var。

在训练阶段，running_mean和running_var在每次前向时更新一次；

在测试阶段，则通过net.eval()固定该BN层的running_mean和running_var，**此时这两个值即为训练阶段最后一次前向时确定的值**，并在整个测试阶段保持不变



rx+b  

Clean: r1x+b1

Noise:r2x+b2

(r1+r2)x+b1+b2





从PyTorch 0.4.1开始, BN层中新增加了一个参数 track_running_stats,

BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)

这个参数的作用如下:

训练时用来统计训练时的forward过的min-batch数目,每经过一个min-batch, track_running_stats+=1
如果没有指定momentum, 则使用1/num_batches_tracked 作为因数来计算均值和方差(running mean and variance).

![image-20230920112439997](https://s2.loli.net/2024/03/08/Teg5Dryo1Id92A7.png)





`track_running_stats==False`

可以使用这个来控制，running_mean和running_var是否统计