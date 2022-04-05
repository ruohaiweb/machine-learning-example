# -*-coding:utf-8-*-
import csv
import pickle
import re

from torch.utils.data import Dataset

from vocab import Vocab

data_path = r"data/predict_data.csv"

Vocab()


class ImdbDataset(Dataset):
    def __init__(self, sequence_max_len):
        super(ImdbDataset, self).__init__()
        self.sequence_max_len = sequence_max_len
        self.data_list = []
        with open(data_path, "r", encoding="gbk") as file:
            self.data_list = list(csv.reader(file))
        self.voc_model = pickle.load(open("./models/vocab.pkl", "rb"))

    def __getitem__(self, idx):
        content = self.data_list[idx + 1]
        if len(content) == 1:
            content_tokens = tokenlize(content[0])
        else:
            content_tokens = tokenlize("")
        voc_result = self.voc_model.transform(content_tokens, max_len=self.sequence_max_len)
        return voc_result

    # file = self.total_path[idx]
    # # 从txt获取评论并分词
    # review = tokenlize(open(file, "r", encoding="utf-8").read())
    # # 获取评论对应的label
    # label = int(file.split("_")[-1].split(".")[0])
    # label = 0 if label < 5 else 1

    # return review, label

    def __len__(self):
        # return len(self.total_path)
        return len(self.data_list) - 1


def tokenlize(sentence):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """

    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<br />", " ", sentence)
    # sentence = re.sub("I'm","I am",sentence)
    # sentence = re.sub("isn't","is not",sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]

    return result


def get_data_list():
    with open(data_path, "r", encoding="gbk") as file:
        content_list = list(csv.reader(file))

    return content_list


if __name__ == "__main__":
    # data_path = r"./data/predict_data.csv"
    # data_list = []
    # with open(data_path, "r", encoding="gbk") as file:
    #     data_list = list(csv.reader(file))
    #     print(data_list)
    get_data_list()
