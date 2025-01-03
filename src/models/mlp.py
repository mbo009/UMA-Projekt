import torch
import torch.nn as nn
import torch.optim as optim
from models.model import Model


class MLP(Model):
    def __init__(
        self,
        train_epochs=100,
        input_size=30,
        hidden_size1=64,
        hidden_size2=32,
        output_size=1,
    ):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
            nn.Sigmoid(),
        )
        self.epochs = train_epochs
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCELoss()

    def forward(self, X):
        return self.model(X)

    def train(self, X, y):
        for _ in range(self.epochs):
            outputs = self.model(X)
            loss = self.criterion(outputs, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        predictions = self.model(X)
        return predictions.detach().cpu().numpy().flatten()
