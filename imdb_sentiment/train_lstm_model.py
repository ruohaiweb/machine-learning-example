import torch

from imdb_sentiment import imdb_lstm_model

if __name__ == '__main__':
    imdb_dataset = imdb_lstm_model.get_dataset()
    # 训练
    imdb_model = imdb_lstm_model.ImdbModel(imdb_dataset.get_num_embeddings(), imdb_dataset.get_padding_idx()).to(imdb_lstm_model.device())
    imdb_lstm_model.train(imdb_model, imdb_dataset, 6)
    # 测试
    # 必须先引入这个类：from imdb_lstm_model import ImdbModel
    from imdb_lstm_model import ImdbModel
    path_fc_model = "./models/lstm_model.pkl"
    fc_model = torch.load(path_fc_model)
    imdb_lstm_model.test(fc_model, imdb_dataset)
    # predict.predict_lstm_model()
