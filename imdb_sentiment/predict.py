import csv
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vocab import Vocab
Vocab()
from imdb_sentiment import dataset_predict, imdb_lstm_model, imdb_fc_model


# voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
# Vocab()


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def predict(model, data_list):
    model.eval()
    result_list = []
    with torch.no_grad():
        for data in tqdm(data_list):
            data = data.to(device())
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            result_list.extend(pred.numpy().tolist())
    new_result_list = []
    for result in result_list:
        new_result_list.append(result[0])
    return new_result_list


def export_result(result_list):
    data_list = dataset_predict.get_data_list()

    data_result_list = list(zip(data_list, result_list))
    print(data_result_list)
    now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    with open(f"./data/result/data_0315_result_{now}.csv", "w", encoding="utf-8-sig", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data_result_list)


def collate_fn2(batch):
    """
    对batch数据进行处理
    :param
    :return: 元组
    """

    # reviews = torch.LongTensor([voc_model.transform(i, max_len=sequence_max_len) for i in batch])
    reviews = torch.LongTensor(batch)
    return reviews


def get_dataloader2(sequence_max_len):
    imdb_dataset = dataset_predict.ImdbDataset(sequence_max_len)
    batch_size = 500
    return DataLoader(imdb_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn2)


def predict_fc_model():
    path_fc_model = "./models/fc_model.pkl"
    fc_model = torch.load(path_fc_model)
    sequence_max_len = imdb_fc_model.sequence_max_len
    data_list = get_dataloader2(sequence_max_len)
    result_list = predict(fc_model, data_list)
    export_result(result_list)


def predict_lstm_model():

    # ImdbModel()
    path_fc_model = "./models/lstm_model.pkl"
    fc_model = torch.load(path_fc_model)
    sequence_max_len = imdb_lstm_model.sequence_max_len
    data_list = get_dataloader2(sequence_max_len)
    result_list = predict(fc_model, data_list)
    export_result(result_list)


if __name__ == '__main__':
    # 必须先引入这个类：from imdb_fc_model import ImdbModel
    # from imdb_fc_model import ImdbModel
    # predict_fc_model()
    # 必须先引入这个类：from imdb_lstm_model import ImdbModel
    from imdb_lstm_model import ImdbModel
    predict_lstm_model()
