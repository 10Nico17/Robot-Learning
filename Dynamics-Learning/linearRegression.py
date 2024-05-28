import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_data=200):
    env = gym.make('MountainCarContinuous-v0', render_mode='human')

    # for this problem observation=state
    u_dim = env.action_space.shape[0]
    x_dim = env.observation_space.shape[0]
    data_input = []
    data_target = []

    dynData_state = []
    dynData_acc = []
    dynData_control = []

    x_state, info = env.reset()

    for t in range(n_data):
        # u_controls = env.action_space.sample()  # agent policy that uses the observation and info
        freq = 1+t//100
        u_controls = np.sin([.01*freq*t])

        x_prev = x_state
        x_state, reward, terminated, truncated, info = env.step(u_controls)

        data_input.append(np.concatenate([x_prev, u_controls]))
        data_target.append(x_state)

        dynData_state.append(x_prev)
        dynData_control.append(u_controls)
        dynData_acc.append(x_state[1:2]-x_prev[1:2])

        if terminated or truncated:
            # terminated = a terminal state (often goal state, sometimes kill state, typically with pos/neg reward) is reached; formally: the infinite MPD transitions to a deadlock nirvana state with eternal zero rewards
            # truncated = the simulation is 'artificially' truncated by some time limited - that's actually formally inconsistent to the definition of an infinite MDP
            if truncated:
                print('-- truncated -- should not happen!')
            else:
                print('-- terminated -- goal state reached')
            x_state, info = env.reset()

    env.close()

    np.savez('data.npz', input=data_input, target=data_target)
    np.savez('dynData.npz', state=dynData_state, controls=dynData_control, acc=dynData_acc)


def train_LinearRegression(invDynamics=False):
    if not invDynamics:
        fil = np.load('data.npz')
        data_input = fil['input']
        data_target = fil['target']
    else:
        fil = np.load('dynData.npz')
        data_input = np.hstack((fil['state'], fil['acc']))
        data_target = fil['controls']
    print(f'== loaded data: input: {data_input.shape}, target: {data_target.shape}')

    # randomly shuffle the data
    ALL = np.hstack((data_input, data_target))
    np.random.shuffle(ALL)
    X, Y = np.hsplit(ALL, [data_input.shape[1]])

    # build features
    n = X.shape[0]
    X = np.hstack((np.ones((n,1)), X, np.cos(3.*X[:,0:1])))

    # split into test and train
    n = n - n//10
    print(f'== data split in train {n} and test {X.shape[0]-n}')
    X_train, Y_train = X[:n,:], Y[:n,:]
    X_test, Y_test = X[n:,:], Y[n:,:]

    # compute linear regression
    lambda_ = 0 #1e-10
    theta = np.linalg.inv(X_train.T @ X_train + lambda_ * np.eye(X.shape[1])) @ X_train.T @ Y_train

    np.set_printoptions(precision=6, suppress=True)
    print('== linear regression: theta=\n', theta)
    print('== ground truth dynamics: v\' = v + 0.0015*u - 0.0025*cos(3x) (=> gym is not doing leap frog, but naive Euler)')
    print('== training error: mse=', np.sum((Y_train - X_train@theta)**2)/n)
    print('== test error:     mse=', np.sum((Y_test - X_test@theta)**2)/n)

    return theta

def control(theta, q_des, T):
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    x_state, info = env.reset()

    tau = 3
    xi = 0.9 # for .5 you'll see an overshooting
    kp = 1/(tau**2)
    kd = 2*xi/tau
    trajectory = []

    for t in range(T):
        q = x_state[0]
        v = x_state[1]
        acc_des = kp * (q_des - q) - kd * v
        input = np.array([1, q, v, acc_des, np.cos(3.*q)])
        u_controls = input.T@theta
        x_state, reward, terminated, truncated, info = env.step(u_controls)

        trajectory.append(x_state)

        if terminated or truncated:
            if truncated:
                print('-- truncated -- should not happen!')
            else:
                print('-- terminated -- goal state reached')
            x_state, info = env.reset()

    env.close()

    trajectory = np.vstack(trajectory)
    fig, ax = plt.subplots()
    ax.plot(range(1,T+1), trajectory[:,0], label='pos')
    ax.plot(range(1,T+1), trajectory[:,1], label='vel')
    plt.legend(loc='best')
    plt.show()


generate_data(400)

exercise='d'

if exercise=='b':
    theta = train_LinearRegression(False)
    
elif exercise=='d':
    theta = train_LinearRegression(True)
    control(theta, -.45, 100)
