import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


def expert(t):
    if t < 50:
        return np.array([-1.0])
    elif t < 100:
        return np.array([1.0])
    return np.array([0.0])

def generate_data(output_file, n_data=10000):
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    u_dim = env.action_space.shape[0]
    x_dim = env.observation_space.shape[0]
    data_input = np.zeros((0,x_dim))
    data_target = np.zeros((0,u_dim))
    n_data = 10000

    x_state, info = env.reset()
    t = 0

    for _ in range(n_data):
        # u_controls = env.action_space.sample()  # agent policy that uses the observation and info
        u_controls = expert(t)
        x_prev = x_state
        x_state, reward, terminated, truncated, info = env.step(u_controls)
        t = t + 1
        # terminated = a terminal state (often goal state, sometimes kill state, typically with pos/neg reward) is reached;
        #    formally: the infinite MPD transitions to a deadlock nirvana state with eternal zero rewards
        # truncated = the simulation is 'artificially' truncated by some time limited - that's actually formally inconsistent to the definition of an infinite MDP

        data_input = np.vstack([data_input, x_prev])
        data_target = np.vstack([data_target, u_controls])

        if terminated or truncated:
            if truncated:
                print('-- truncated -- should not happen!')
            else:
                print('-- terminated -- goal state reached')
            x_state, info = env.reset()
            t = 0

    env.close()

    print('input data:', data_input.shape)
    print('output data:', data_target.shape)

    np.savez_compressed(output_file, data_input=data_input, data_target=data_target)

def load_data(file):
    with np.load(file) as data:
        data_input = data['data_input']
        data_target = data['data_target']

    fig, ax = plt.subplots()
    ax.scatter(data_input[:,0], data_input[:,1],c=data_target)
    plt.show()

    return data_input, data_target

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.seq(x)

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

def get_device():
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def imitation_learning(file, output_file):
    data_input, data_target = load_data(file)

    ## Solution, based on https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    data_input = torch.from_numpy(data_input).float()
    data_target = torch.from_numpy(data_target).float()

    dataset = TensorDataset(data_input, data_target)
    training_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    device = get_device()
    model = NeuralNetwork()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    torch.save(model.state_dict(), output_file)
    print("Saved PyTorch Model State")

def validation(model_file):
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    device = get_device()
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))

    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    x_state, info = env.reset(options={'low': 0.1, 'high': 0.4})
    # x_state = env.state = np.array([-0.5,0.3])
    while True:
        # u_controls = env.action_space.sample()  # agent policy that uses the observation and info
        print(torch.from_numpy(x_state).float().reshape(1,2))
        u_controls = model(torch.from_numpy(x_state).float().reshape(1,2)).detach().numpy()
        print(u_controls)
        x_prev = x_state
        x_state, reward, terminated, truncated, info = env.step(u_controls)

        if terminated or truncated:
            if truncated:
                print('-- truncated -- should not happen!')
            else:
                print('-- terminated -- goal state reached')
            break
    env.close()

def main():
    generate_data("data_expert1.npz")
    imitation_learning("data_expert1.npz", "model.pth")
    validation("model.pth")


if __name__ == '__main__':
    main()