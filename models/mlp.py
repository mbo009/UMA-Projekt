import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size=30, hidden_size1=64, hidden_size2=32, output_size=1):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
            nn.Sigmoid(),
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def train(self, x, y, epochs=100):
        for _ in range(epochs):
            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    model = MLP()
    print(model)
    x = torch.randn(10, 30)
    y = torch.randint(0, 2, (10, 1)).float()
    model.train(x, y)
