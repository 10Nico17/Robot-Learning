import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


def expert1(t):
    if t < 50:
        return np.array([-1.0])
    elif t < 100:
        return np.array([1.0])
    return np.array([0.0])

def expert2(t):
    if t < 50:
        return np.array([1.0])
    elif t < 100:
        return np.array([0.0]) # save some energy!
    elif t < 150:
        return np.array([1.0])
    return np.array([1.0])

def generate_data(expert, output_file, n_data=10000):
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

def load_data(files):
    with np.load(files[0]) as data:
        data_input = data['data_input']
        data_target = data['data_target']

    for file in files[1:]:
        with np.load(file) as data:
            data_input = np.concatenate([data_input, data['data_input']])
            data_target = np.concatenate([data_target, data['data_target']])

    fig, ax = plt.subplots()
    ax.scatter(data_input[:,0], data_input[:,1],c=data_target)
    plt.show()

    return data_input, data_target

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

# Define net
class Net(nn.Module):
  def __init__(self, nhidden: int = 256):
    super().__init__()
    layers = [nn.Linear(4, nhidden)] # condition (2) + output (1) + time (1)
    for _ in range(2):
      layers.append(nn.Linear(nhidden, nhidden))
    layers.append(nn.Linear(nhidden, 1)) # output u
    self.linears = nn.ModuleList(layers)

    # init using kaiming
    for layer in self.linears:
      nn.init.kaiming_uniform_(layer.weight)

  def forward(self, x, u, t):
    x = torch.concat([x, u, t], axis=-1)
    for l in self.linears[:-1]:
      x = nn.ReLU()(l(x))
    return self.linears[-1](x)


def get_alpha_betas(N: int):
  """Schedule from the original paper.
  """
  beta_min = 0.1
  beta_max = 20.
  betas = np.array([beta_min/N + i/(N*(N-1))*(beta_max-beta_min) for i in range(N)])
  alpha_bars = np.cumprod(1 - betas)
  return alpha_bars, betas


def train(loader, device, nepochs: int = 10, denoising_steps: int = 100):
  """Alg 1 from the DDPM paper"""
  model = Net()
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  alpha_bars, _ = get_alpha_betas(denoising_steps)      # Precompute alphas
  losses = []
  for epoch in range(nepochs):
    for state, action in loader:
      state = state.to(device)
      action = action.to(device)
      optimizer.zero_grad()

      # Fwd pass
      t = torch.randint(denoising_steps, size=(action.shape[0],))  # sample timesteps - 1 per datapoint
      alpha_t = torch.index_select(torch.Tensor(alpha_bars), 0, t).unsqueeze(1).to(device)    # Get the alphas for each timestep
      noise = torch.randn(*action.shape, device=device)   # Sample DIFFERENT random noise for each datapoint
      model_in = alpha_t**.5 * action + noise*(1-alpha_t)**.5   # Noise corrupt the data (eq14)
      out = model(state, model_in, t.unsqueeze(1).to(device))
      loss = torch.mean((noise - out)**2)     # Compute loss on prediction (eq14)
      losses.append(loss.detach().cpu().numpy())

      # Bwd pass
      loss.backward()
      optimizer.step()

    if (epoch+1) % 100 == 0:
        mean_loss = np.mean(np.array(losses))
        losses = []
        print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))

  return model


def sample(model, state, device, n_samples: int = 50, n_steps: int=100):
  """Alg 2 from the DDPM paper."""
  state = state.repeat(n_samples,1)
  x_t = torch.randn((n_samples, 1)).to(device)
  alpha_bars, betas = get_alpha_betas(n_steps)
  alphas = 1 - betas
  for t in range(len(alphas))[::-1]:
    ts = t * torch.ones((n_samples, 1)).to(device)
    ab_t = alpha_bars[t] * torch.ones((n_samples, 1)).to(device)  # Tile the alpha to the number of samples
    z = (torch.randn((n_samples, 1)) if t > 1 else torch.zeros((n_samples, 1))).to(device)
    model_prediction = model(state, x_t, ts)
    x_t = 1 / alphas[t]**.5 * (x_t - betas[t]/(1-ab_t)**.5 * model_prediction)
    x_t += betas[t]**0.5 * z

  return x_t

def imitation_learning(files, output_file):
    data_input, data_target = load_data(files)
    data_input = torch.from_numpy(data_input).float()
    data_target = torch.from_numpy(data_target).float()

    batch_size = 64
    dataset = TensorDataset(data_input, data_target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = get_device()
    model = train(loader, device, 300)
    torch.save(model.state_dict(), output_file)


def validation(model_file):
    device = get_device()
    model = Net().to(device)
    model.load_state_dict(torch.load(model_file))

    # fig, ax = plt.subplots()
    # ax.scatter(data_input[:,0], data_target[:,0],c=data_target)
    # print(model(data_input).detach().numpy())
    # ax.scatter(data_input[:,0], model(data_input).detach().numpy())

    samples = sample(model, torch.from_numpy(np.array([-1.0,0.0])).float(), device).detach().cpu().numpy()
    print(samples)
    plt.figure(figsize=(5,5))
    plt.hist(samples)
    plt.show()

    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    x_state, info = env.reset()#options={'low': 0.1, 'high': 0.4})
    # x_state = env.state = np.array([-0.5,0.3])
    t=0
    while True:
        # u_controls = env.action_space.sample()  # agent policy that uses the observation and info
        print(x_state)
        u_controls = sample(model, torch.from_numpy(x_state).float(), device, n_samples=1).detach().cpu().numpy().flatten()
        print(u_controls)
        x_state, reward, terminated, truncated, info = env.step(u_controls)
        t=t+1

        if terminated or truncated:
            if truncated:
                print('-- truncated -- should not happen!')
            else:
                print('-- terminated -- goal state reached')
            break
            # x_state, info = env.reset()
            t=0
    env.close()

def main():
    generate_data(expert1, "data_expert1.npz", 5000)
    generate_data(expert2, "data_expert2.npz", 5000)
    imitation_learning(["data_expert1.npz", "data_expert2.npz"], "model.pth")
    validation("model.pth")


if __name__ == '__main__':
    main()