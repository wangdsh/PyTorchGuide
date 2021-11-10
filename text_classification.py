import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

start_time = time.time()

class NN(nn.Module):
    """模型
    """
    def __init__(self, vocab_size, emb_dim, padding_idx, class_num):
        super(NN, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx)
        self.fc = nn.Linear(emb_dim, class_num)

    def forward(self, data):
        emb = self.emb(data)
        emb = emb.mean(dim=1)
        out = self.fc(emb)

        return out

tokenizer = get_tokenizer('basic_english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter 模型相关参数
batch_size = 32
num_epochs = 3
min_freq = 5
emb_dim = 64
class_num = 4

class Vocab:
    """词汇表
    """
    def __init__(self, min_freq):
        self.min_freq = min_freq
        self.itos = {0:'<PAD>', 1:'<SOS>', 2:'<EOS>', 3:'<UNK>'}
        self.stoi = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, sent_iter):
        freq = {}
        idx = 4 # 0-3已被占用

        for sentence in sent_iter:
            for word in tokenizer(sentence):
                freq[word] = 1 if word not in freq else freq[word] + 1

                if freq[word] == self.min_freq:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def digitize(self, sent):
        """string -> number
        """
        sent = [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in sent]
        sent = [self.stoi['<SOS>']] + sent + [self.stoi['<EOS>']] # 添加句首、句尾标识

        return sent

class SentIter:
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.train_dataset, None)
        if item:
            return item[1]

        item = next(self.test_dataset, None)
        if item:
            return item[1]

        raise StopIteration

# 获取所有句子，以构建词典
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='dataset/', split=('train', 'test'))
sent_iter = SentIter(train_dataset, test_dataset)

vocab = Vocab(min_freq)
vocab.build_vocab(sent_iter)

class MyCollate:
    """批数据转换为可输入模型的格式
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        labels = [item[0]-1 for item in batch] # 1-4 -> 0-3
        texts = [torch.tensor(vocab.digitize(tokenizer(item[1]))) for item in batch]
        texts = pad_sequence(texts, batch_first=True, padding_value=self.pad_idx)

        return texts.to(device), torch.tensor(labels, device=device)

# 加载数据
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='dataset/', split=('train', 'test'))
train_dataset = list(train_dataset) # 非迭代器模式下才可以运行多个epoch
test_dataset = list(test_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, \
                          collate_fn=MyCollate(vocab.stoi['<PAD>']))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, \
                          collate_fn=MyCollate(vocab.stoi['<PAD>']))

# 创建模型
model = NN(len(vocab), emb_dim, vocab.stoi['<PAD>'], class_num).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Train network
for epoch in range(num_epochs):
    total_acc, total_cnt = 0, 0
    print('epoch:', epoch)

    for batch_idx, (text, label) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(text)

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        total_acc += (pred.argmax(dim=1) == label).sum().item()
        total_cnt += label.size(0)

        if batch_idx % 200 == 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:6.3f} |'.format(epoch, \
                    batch_idx, len(train_loader), total_acc/total_cnt))

# 评估
model.eval()
total_acc, total_cnt = 0, 0
with torch.no_grad():
    for batch_idx, (text, label) in enumerate(test_loader):
        pred = model(text)
        total_acc += (pred.argmax(dim=1) == label).sum().item()
        total_cnt += label.size(0)
print('\n| test accuracy: {:6.3f} |'.format(total_acc/total_cnt))

end_time = time.time()
run_time = round(end_time - start_time, 2)
print(f'\nDone. Running time: {run_time} seconds.')

