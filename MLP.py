import copy
import math
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.optim import Adam


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class _MLP_Module(nn.Module):

    def __init__(self, input_size, layer_sizes):
        super().__init__()
        self.__layer_sizes = layer_sizes

        self.in_layer = nn.Linear(input_size, layer_sizes[0])
        self.hidden = nn.ModuleList()
        for k in range(len(layer_sizes) - 1):
            self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
        self.out = nn.Linear(layer_sizes[-1], 2)

    def forward(self, x):
        x = torch.relu(self.in_layer(x))
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return torch.softmax(self.out(x), dim=1)


class MLP_Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, input_size=100, layer_sizes=(100,), learning_rate=0.01, n_epochs=10, scoring=None):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.scoring = scoring

        self.__module = _MLP_Module(self.input_size, self.layer_sizes)
        self.__optimizer = Adam(self.__module.parameters(), lr=learning_rate)
        self.__best_model_state_dict = {}

    def fit(self, x, y, sample_weight=None):
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()

        if sample_weight:
            criterion = nn.BCELoss(weight=torch.tensor(sample_weight))
        else:
            criterion = nn.BCELoss()
        best_score = math.inf

        self.__module.train()

        for epoch in range(self.n_epochs):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            self.__optimizer.zero_grad()
            train_pred = self.__module.forward(x)
            train_loss = criterion(train_pred[:, 1], y.float())
            train_loss.backward()
            self.__optimizer.step()
            accuracy = accuracy_score(y.detach().numpy(), train_pred.argmax(dim=1))

            if train_loss.item() < best_score:
                self.__best_model_state_dict = copy.deepcopy(self.__module.state_dict())
                best_score = train_loss.item()

            if (epoch + 1) % 500 == 0:
                print(f"epoch #{epoch + 1} train_loss: {round(train_loss.item(), 5)}, train_accuracy:{accuracy}")

    def predict(self, x):
        x = torch.tensor(x).float()
        self.__module.load_state_dict(self.__best_model_state_dict)
        self.__module.eval()
        x = x.to(DEVICE)
        pred_probs = self.__module.forward(x)
        self.__module.train()
        return pred_probs.argmax(dim=1).detach().numpy()

