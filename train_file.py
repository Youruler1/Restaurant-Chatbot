import torch
import torch.nn as nn

def model_setup(input_dim, output_dim, hidden_dim):
    class Model(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim):
            super().__init__()
            self.l1 = nn.Linear(input_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
        def forward(self, x):
            x = self.l1(x)
            x = self.relu(x)
            x = self.l2(x)
            x = self.relu(x)
            return x
    model = Model(input_dim, output_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    return model, optimizer, loss_fn  

# train and target lists must be converted into pytorch tensors
def train_fn(train, target, input_dim, output_dim, num_epochs = 200):
    train = torch.as_tensor(train, dtype = torch.float32)
    target = torch.as_tensor(target, dtype = torch.int64)
    model, optimizer, loss_fn = model_setup(input_dim, output_dim, hidden_dim = 16)
    for epoch in range(num_epochs):
        output = model(train)
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(loss.cpu().detach().numpy())
    return model