# -*-coding:utf-8-*-
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset_train
from vocab import Vocab

train_batch_size = 512
test_batch_size = 500
# voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
sequence_max_len = 100
Vocab()


def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)
    # reviews = torch.LongTensor([voc_model.transform(i, max_len=sequence_max_len) for i in reviews])
    reviews = torch.LongTensor(reviews)
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataset():
    return dataset_train.ImdbDataset(train)


def get_dataloader(imdb_dataset, train=True):
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

class ImdbModel(nn.Module):
    def __init__(self, num_embeddings, padding_idx):
        super(ImdbModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=200, padding_idx=padding_idx)

        self.fc = nn.Linear(sequence_max_len * 200, 2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """
        input_embeded = self.embedding(input)  # input embeded :[batch_size,max_len,200]

        # 变形
        input_embeded_viewed = input_embeded.view(input_embeded.size(0), -1)

        # 全连接
        out = self.fc(input_embeded_viewed)
        return F.log_softmax(out, dim=-1)


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train(imdb_model, imdb_dataset, epoch):
    """

    :param imdb_model:
    :param epoch:
    :return:
    """
    train_dataloader = get_dataloader(imdb_dataset,train=True)
    # bar = tqdm(train_dataloader, total=len(train_dataloader))

    optimizer = Adam(imdb_model.parameters())
    for i in range(epoch):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, (data, target) in enumerate(bar):
            optimizer.zero_grad()
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))
    # 保存模型
    path_model = "./models/fc_model.pkl"
    torch.save(imdb_model, path_model)
    # 保存模型参数
    path_state_dict = "./models/fc_model_state_dict.pkl"
    net_state_dict = imdb_model.state_dict()
    torch.save(net_state_dict, path_state_dict)


def test(imdb_model, imdb_dataset):
    """
    验证模型
    :param imdb_model:
    :return:
    """
    test_loss = 0
    correct = 0
    imdb_model.eval()
    test_dataloader = get_dataloader(imdb_dataset,train=False)
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
