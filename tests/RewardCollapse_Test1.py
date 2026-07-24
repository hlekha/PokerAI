#!/usr/bin/env python
# coding: utf-8

# In[ ]:


env = Poker()

is_ipython = 'inline' in matplotlib.get_backend() # checking if program is run in a notebook or command terminal
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

PokerExperience = namedtuple("PokerExperience", ('observation', 'action', "next_obs", 'reward'))

class MemoryBank(object):
    """
    Replay Buffer!!! 
    """

    def __init__(self, bound):
        self.memory =  deque([], maxlen = bound)

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """
        Notes for push function:
        Adds the new experience to the memory bank
        *args lets the function accept any number of parameters
        """
        self.memory.append(PokerExperience(*args))

    def sample(self, size):
        return random.sample(self.memory, size)

def select_action(obs):
    """
    Notes for select_action function:
    Selects the action using an epsilon-greedy policy, where epsilon has a decay rate.
    This leads to high learning in the beginning, but high exploration in the end.
    """
    global hands
    n = random.random()
    eps_upper = EPS_FINAL + (EPS_0 - EPS_FINAL) * math.exp(-1.0 * hands / EPS_DECAY)
    hands += 1
    if n > eps_upper:
        with torch.no_grad():
            q_values = policy_net(obs)
            mask = torch.tensor(env.action_mask(), device = device)
            q_values[:,~mask] = -1e9

            return q_values.argmax(1).view(1,1)
    else:
        legal = np.where(env.action_mask())[0]
        action = random.choice(legal)
        return torch.tensor([[action]], device = device, dtype = torch.long)


class PokerDQN(nn.Module):

    def __init__(self, n_obs, n_actions):
        """
        Notes for __init__ function:
        Establishes layers for the network, and the amount of nodes in each layer
        with its activation function reflecting the ReLU function. 
        Prob don't have a dying ReLU, should experiment with a leaky ReLU.
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, n_actions)
        )


    def forward(self, x):

        return self.network(x)


#---------------------------------------------------------------------------------
# Hyper and regular parameters
#---------------------------------------------------------------------------------

BATCH = 256
GAMMA = 0.97
EPS_0 = 0.9
EPS_FINAL = 0.01
EPS_DECAY = 300
TAU = 0.005
ALPHA = 3e-4

n_actions = 7
observation, info = env.reset()
n_obs = len(observation)    

policy_net = PokerDQN(n_obs, n_actions).to(device) 
target_net = PokerDQN(n_obs, n_actions).to(device) 
target_net.load_state_dict(policy_net.state_dict()) 

optimizer = optim.AdamW(policy_net.parameters(), lr = ALPHA, amsgrad = True) 
memory = MemoryBank(275000)
hands = 0



reward_episodes = []
def plot_reward(show_result = False):
    """
    Notes for plot_reward function:
    This is for plotting the result of each episode,
    and creating a simple moving average line of the rewards.
    """
    reward_t = torch.tensor(reward_episodes, dtype = torch.float)

    plt.figure(1, figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.cla()

    if show_result:
        plt.title('Done')

    else:
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.scatter(range(len(reward_t)), reward_t.numpy(), s=5, color='blue')

    if len(reward_t) >= 100:
        means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), color = "green")



loss_history = []
def optimize_model():
    """
    Notes for optimize_model function:
    Loss function and gradient step
    """
    if len(memory) < BATCH:
        return
    transitions = memory.sample(BATCH)
    batch = PokerExperience(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_obs)), device=device, dtype=torch.bool)
    non_final_next_obs = torch.cat([s for s in batch.next_obs if s is not None])
    obs_batch = torch.cat(batch.observation)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    obs_action_values = policy_net(obs_batch).gather(1, action_batch)

    next_obs_values = torch.zeros(BATCH, device=device)
    with torch.no_grad():
        best_actions = (policy_net(non_final_next_obs).argmax(dim=1, keepdim=True))

        next_obs_values[non_final_mask] = (target_net(non_final_next_obs).gather(1, best_actions).squeeze(1))

    expected_obs_action_values = (next_obs_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(obs_action_values, expected_obs_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    loss_history.append(loss.item())


def plot_loss(show_result=False):
    loss_t = torch.tensor(loss_history, dtype=torch.float)

    plt.figure(1, figsize=(12,5))
    plt.subplot(1, 2, 2)
    plt.cla()

    if show_result:
        plt.title("Loss Curve is done")
    else:
        plt.clf()
        plt.title("Loss Curve loading...")

    plt.xlabel("Gradient Step")
    plt.ylabel("Loss")

    if len(loss_t) > 100:
        mean = loss_t.unfold(0, 100, 1).mean(1).view(-1)
        mean = torch.cat((torch.zeros(99), mean))
        plt.plot(mean.numpy(), color = "red")

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

#-----------------------------------------------------------------------------------------------------
# Actual training
#-----------------------------------------------------------------------------------------------------

num_episodes = 500

for episode in range(num_episodes):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0

    for t in count():
        action = select_action(obs)
        current_obs = obs
        obs, reward, terminated, truncated, info = env.step(action.item())

        reward = reward / env.stack         
        reward = torch.tensor([reward], dtype=torch.float32, device=device)

        done = terminated or truncated

        if terminated:
            next_obs = None
        else:
            next_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(current_obs, action, next_obs, reward)
        obs = next_obs

        optimize_model()

        target_net_obs_dict = target_net.state_dict()
        policy_net_obs_dict = policy_net.state_dict()
        for key in policy_net_obs_dict:
            target_net_obs_dict[key] = policy_net_obs_dict[key]*TAU + target_net_obs_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_obs_dict)
        total_reward += reward.item()
        if done:
            reward_episodes.append(total_reward)
            plot_reward()
            plot_loss()
            break
weights = torch.save(policy_net.state_dict(), "policy_net_weights.pth")
print('Complete')
plot_reward(show_result=True)
plot_loss(show_result=True)
plt.ioff()
plt.show()

