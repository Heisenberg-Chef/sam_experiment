import torch
import torch.nn as nn

# 定义一个简单的模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = MyModel()

# 冻结部分权重
for param in model.parameters():
    param.requires_grad = True

# 定义输入
input_data = torch.randn(1, 10)

# 前向传播
output = model(input_data)

# 计算损失（这里只是一个例子，损失值不重要）
loss = torch.sum(output)

# 反向传播
loss.backward()

# 打印权重和梯度信息
for name, param in model.named_parameters():
    print(f"{name}:")
    print(f"  requires_grad: {param.requires_grad}")
    print(f"  grad_fn: {param.grad_fn}")
    print(f"  grad: {param.grad}")
    print()
