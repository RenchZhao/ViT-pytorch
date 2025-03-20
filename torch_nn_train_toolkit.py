import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch import nn
from torch.nn import functional as F
import matplotlib
import matplotlib.pyplot as plt
import os

def load_data_mnist(batch_size, resize=None):
    """下载MNIST数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor(),
             transforms.Normalize(0.1307, 0.3081)]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root="./data", 
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.MNIST(root="./data",
                                                   train=False,
                                                  transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))

def load_data_CIFAR10(batch_size, resize=None, data_agmt=False):
    """下载CIFAR10数据集，然后将其加载到内存中。"""
    if data_agmt:
        trans = [transforms.RandomCrop(32,padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914,0.4822,0.4465),
                                      (0.2470,0.2435,0.2616))]
    else:
        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.4914,0.4822,0.4465),
                                      (0.2470,0.2435,0.2616))]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    CIFAR10_train = torchvision.datasets.CIFAR10(root="./data", 
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    CIFAR10_test = torchvision.datasets.CIFAR10(root="./data",
                                                   train=False,
                                                  transform=trans,
                                                   download=True)
    return (data.DataLoader(CIFAR10_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(CIFAR10_test, batch_size, shuffle=False,
                            num_workers=0))

def train_modl(net, train_iter, test_iter, num_epochs, loss, lr, device, optimizer=None, scheduler=None, init=True, tr_hist_ls=[[],[],[],[]]):
    """用GPU训练模型，参考《动手学深度学习》-李沐"""
    def init_weights(m):
        """初始化模型权重"""
        if type(m) == nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            
    def accuracy(y_hat,y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype)== y
        return float(cmp.type(y.dtype).sum())

    def stat_test_loss_acc(net, test_iter, loss_f, device=None):
        """计算测试集中预测正确的数量"""
        acc_num=0
        loss_num=0
        if isinstance(net, torch.nn.Module):
            net.eval() # 设置为评估模式
            if not device:
                device = next(iter(net.parameters())).device
        for X, y in test_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            try:
                with torch.no_grad():
                    y_hat=net(X)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            acc_num += accuracy(y_hat,y)
            loss_num += (loss_f(y_hat,y)* X.shape[0])
        return loss_num, acc_num

    def show_train_history(train_ls, test_ls, eva_type, fname='output.jpg'):
        """展示训练集/测试集准确率/损失曲线"""
        plt.plot(train_ls)
        plt.plot(test_ls)
        plt.title('Train History')
        plt.ylabel(eva_type)
        plt.xlabel('Epoch')
        plt.legend(['train {}'.format(eva_type), 'test {}'.format(eva_type)], loc='upper left')
        plt.savefig(fname)
        plt.show()
    
    if init:
        try:
            net.apply(init_weights)
        except:
            print("Error! Failed to initialize model weight.")
        else:
            print("initialized model weight")
    else:
        print("Did not initialize model weight")
    
    print('training on', device)
    net.to(device)
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    result_ls=[0,0,0,0]     # result_ls[0]:total train loss
                            # result_ls[1]:train acc sample numbers
                            # result_ls[2]:total test loss
                            # result_ls[3]:test acc sample numbers
                            # update per epoch
                            
    num_train_samples=0
    num_test_samples=0
    print("examing the sample numbers of training set/ testing set……")
    for i, (X,y) in enumerate(train_iter):
        num_train_samples+=y.numel()
    for i, (X,y) in enumerate(test_iter):
        num_test_samples+=y.numel()
    print("the sample numbers of training set is {}".format(num_train_samples))
    print("the sample numbers of testing set is {}".format(num_test_samples))
    print("start training……")
    
    num_batches = len(train_iter)
    train_acc = 0
    train_loss = 0
    test_acc = 0
    tr_loss_ls=tr_hist_ls[0]
    tr_acc_ls=tr_hist_ls[1]
    te_loss_ls=tr_hist_ls[2]
    te_acc_ls=tr_hist_ls[3]
    # 这里要改一下，把训练函数封装起来
    for epoch in range(num_epochs):
        net.train()
        result_ls=[0,0,0,0]
        for i, (X,y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            result_ls[0]+= l * X.shape[0]
            result_ls[1]+= accuracy(y_hat,y)
            
        if type(scheduler)!=type(None):
            scheduler.step()
            
        result_ls[2],result_ls[3]=stat_test_loss_acc(net,test_iter ,loss_f=loss)
        train_loss = result_ls[0] / num_train_samples
        train_acc = result_ls[1] / num_train_samples
        test_loss = result_ls[2] / num_test_samples
        test_acc = result_ls[3] / num_test_samples
        tr_loss_ls.append(float(train_loss))
        tr_acc_ls.append(float(train_acc))
        te_loss_ls.append(float(test_loss))
        te_acc_ls.append(float(test_acc))
        print('epoch[{}/{}] loss:{:0<.4f} train acc:{:0<.4f} test loss:{:0<.4f} test acc:{:0<.4f}'.format(
                                                                      epoch+1,
                                                                      num_epochs,
                                                                      float(train_loss),
                                                                      float(train_acc),
                                                                      float(test_loss),
                                                                      float(test_acc)))
        if (epoch+1) % 5==0:
            save_model(net, epoch=len(tr_loss_ls),tr_hist_ls=[tr_loss_ls, tr_acc_ls, te_loss_ls, te_acc_ls],path="BackUp_modl.ckpt")
            
    show_train_history(tr_loss_ls, te_loss_ls, eva_type='loss', fname='loss.jpg')
    show_train_history(tr_acc_ls, te_acc_ls, eva_type='accuracy', fname='accuracy.jpg')
    print("training finished")
    print('test acc:{:0<.4f}'.format(float(test_acc)))
    return [tr_loss_ls, tr_acc_ls, te_loss_ls, te_acc_ls]

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def save_model(model,epoch=0,tr_hist_ls=[[],[],[],[]], path=""):
    if len(path)!=0:
        state = {'model': model.state_dict(), 'epoch': epoch, 'tr_hist_ls': tr_hist_ls}
        torch.save(state, path)
    
def load_model_params(model, epoch=0, tr_hist_ls=[[],[],[],[]], path=""):
    success=False
    if len(path)!=0 and os.path.exists(path):
        print("正在加载模型……")
        try:
            model_CKPT = torch.load(path)
            model.load_state_dict(model_CKPT['model'])
            epoch = model_CKPT['epoch']
            tr_hist_ls = model_CKPT['tr_hist_ls']
            print("上次训练到第{}个epoch".format(epoch))
        except:
            print("加载模型失败！需要从头开始训练。")
        else:
            success=True
            print("加载模型成功！")
    else:
        print("没有找到模型检查点，需要从头开始训练。")
    return model,epoch,tr_hist_ls,success
