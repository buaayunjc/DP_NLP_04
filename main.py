# — coding: utf-8 –
import argparse
import os

import jieba
import torch
import torch.nn as nn
from  tqdm import tqdm

class LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        outputs = self.fc_out(outputs)

        return outputs

def content_deal(path):
    # 语料预处理，进行断句，去除一些广告和无意义内容
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com',
          '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '=', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']

    # 读取文本数据并进行预处理
    dataset = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='gb18030', errors='ignore') as f:
                content = f.read()
                for a in ad:
                    content = content.replace(a, '')
                content=[word for word in jieba.lcut(content)]
                dataset.append(content)
    return dataset

def tensor_to_str(index2word_dict, class_tensor):
    # 将张量转换为字符串
    class_lst = list(class_tensor)
    words = [index2word_dict[int(index)] for index in class_lst]

    # 将列表中的词语连接为一个字符串
    sentence = ''.join(words)
    return sentence




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--sentence_lenth', type=int, default=20, help='sentence lenth')

    args = parser.parse_args()

    path = '金庸小说集'
    novel_dataset = content_deal(path)

    novel_dataset_all = []

    novel_dataset_all.extend(novel_dataset[0])


    word2index = {}

    for word in novel_dataset_all:
        if word not in word2index:
            word2index[word] = len(word2index)

    index2word = {index: word for word, index in word2index.items()}

    # 将中文转换为索引
    novels_index_lst = [word2index[word] for word in novel_dataset_all]

    vocab_size = len(word2index)  # 词典中总共的词数，是文章有多少个不同的词

    max_epoch = args.epoch
    batch_size = args.bs
    learning_rate = args.lr
    sentence_len = args.sentence_lenth
    train_lst = [i for i in range(0, 10000)]

    device = torch.device('cuda')

    model = LSTM(vocab_size, 30, 512, 2, vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(max_epoch)):
        for i in train_lst:
            inputs = torch.tensor([novels_index_lst[j:j + sentence_len] for j in range(i, i + batch_size)]).to(device)
            targets = torch.tensor([novels_index_lst[j + 1:j + 1 + sentence_len] for j in range(i, i + batch_size)]).to(
                device)
            outputs = model(inputs)

            loss = criterion(outputs.view(outputs.size(0) * outputs.size(1), -1), targets.view(-1))
            model.zero_grad()
            loss.backward()
            optimizer.step()


    generate_length = 500  # 测试长度
    test_set = [novels_index_lst[i:i + sentence_len] for i in range(10000, 30000, 2000)]
    target_set = [novels_index_lst[i:i + sentence_len + generate_length] for i in range(10000, 30000, 2000)]

    with torch.no_grad():
        for i in range(0, 2):
            generate_lst = []
            generate_lst.extend(test_set[i])
            for j in range(0, generate_length):
                inputs = torch.tensor(generate_lst[-sentence_len:]).unsqueeze(0)
                outputs = model(inputs)

                predicted_class = torch.argmax(outputs, dim=-1).squeeze(0)

                generate_lst.append(int(predicted_class[-1]))

            input_sentence = tensor_to_str(index2word, test_set[i])
            generate_sentence = tensor_to_str(index2word, generate_lst)
            target_sentence = tensor_to_str(index2word, target_set[i])

            with open("article.txt","w",encoding='gb18030') as file:
                file.write(generate_sentence)
            file.close()


