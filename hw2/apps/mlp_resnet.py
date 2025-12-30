import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim,hidden_dim),
                norm(),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim,dim),
                norm(),
                )
            ),
        nn.ReLU()
        )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim,hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim//2,num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_function=nn.SoftmaxLoss()

    if opt:
        model.train()
    else:
        model.eval()

    for x,y in dataloader:
        y_output=model(x)
        batch_loss=loss_function(y_output,y)
        # ? 为什么不是y_output.shape[0]，根据softmaxloss的计算方法？
        loss+=batch_loss*x.shape[0]
        if opt is not None:
            # 重置梯度是个啥
            opt.reset_grad()
            batch_loss.backward()
            opt.step()
        y=y.numpy()
    # return error, loss
        # 用优化器更新参数



    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
