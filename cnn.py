import torch
from torch import nn
from torch.autograd import Variable
from data_reader import *

class Flatten(nn.Module):
    def forward(self, x):
        N, _, _, _ = x.size()
        return x.view(N, -1)

#LeNet likely cnn
cnn = nn.Sequential(
    nn.Conv2d(1, 6, 3, padding=1),
    nn.ReLU(),
    #nn.BatchNorm2d(6),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, 3),
    nn.ReLU(),
    #nn.BatchNorm2d(16),
    nn.MaxPool2d(2, 2),
    Flatten(),
    nn.Linear(576, 150),
    nn.ReLU(),
    nn.Linear(150, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

#optional fc size
#576*150*100*10
#400*120*84*10

def train(model, loss_fn, optimizer, train_data, train_labels, dtype=torch.FloatTensor,
          num_epochs=1, batch_size=128, print_every=100):
    N = train_data.shape[0]
    num_steps = int(N/batch_size)
    for epoch in range(num_epochs):
        print('Epoch %d / %d'%(epoch + 1, num_epochs))
        model.train()
        for step in range(num_steps):
            indx = np.arange(N) #random generate minibatch
            np.random.shuffle(indx)
            indx = indx[:batch_size]
            batch_data = train_data[indx]
            batch_labels = train_labels[indx]
            x_var = Variable(torch.from_numpy(batch_data).type(dtype).view(batch_size, 1, 28, 28))
            y_var = Variable(torch.from_numpy(batch_labels).type(dtype).long())
            scores = model(x_var)
            loss = loss_fn(scores, y_var)
            if (step + 1)%print_every == 0 or step == (num_steps - 1):
                print('Step %d, loss %f'%(step + 1, loss.data[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def predict(model, test_data, dtype=torch.FloatTensor):
    model.eval()
    N = test_data.shape[0]
    x_var = Variable(torch.from_numpy(test_data).type(dtype).view(N, 1, 28, 28))
    scores = model(x_var)
    y_pred = torch.max(scores, dim=1)[1].data.cpu().numpy()
    return y_pred

#Little test on cnn
#temp = Variable(torch.randn(28, 28))
#print(cnn(temp.view(1, 1, 28, 28)).size())

data, labels = load_data_set('./data/train.csv')
data = data/255 #rescale data to [0, 1]

dtype = torch.cuda.FloatTensor
#split train/test set to test cnn
# train_img = data[:40000]
# train_labels = labels[:40000]
# test_img = data[40000:]
# test_labels = labels[40000:]
#
# cnn.type(dtype)
# loss_fn = nn.CrossEntropyLoss().type(dtype)
# optimizer = torch.optim.RMSprop(cnn.parameters(), lr=0.001)
# train(cnn, loss_fn, optimizer, train_img, train_labels, dtype, 12)
# y_pred = predict(cnn, test_img, dtype)
# accuracy = np.mean(y_pred.reshape(-1) == test_labels)
# print('Test accuracy %f' %(accuracy))

test_data = load_test_set('./data/test.csv')
cnn.type(dtype)
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = torch.optim.RMSprop(cnn.parameters(), lr=0.001)
train(cnn, loss_fn, optimizer, data, labels, dtype, 12)
y_pred = predict(cnn, test_data, dtype)
save_results('./data/results.csv', y_pred)


