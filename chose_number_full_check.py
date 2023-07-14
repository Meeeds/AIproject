import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from collections import defaultdict

class CardGame:
    def __init__(self, number_of_cards):
        self.number_of_cards = number_of_cards
        self.all_states = [(i, j) for i in range(1, number_of_cards+1) 
                                  for j in range(1, number_of_cards+1) 
                                  if i != j]
        self.state_idx = 0
        
    def next(self):
        if self.state_idx >= len(self.all_states):
            raise StopIteration
        self.state = {
            'card_1': self.all_states[self.state_idx][0],
            'card_2': self.all_states[self.state_idx][1],
        }
        self.state_idx += 1
        return self.state

    def last(self):
        return self.state_idx >= len(self.all_states)

    def get_state(self):
        return self.state

    def reset(self):
        self.state_idx = 0
        player_cards = np.random.choice(range(1, self.number_of_cards), size=2, replace=False)
        self.state = {
            'card_1': player_cards[0],
            'card_2': player_cards[1],
        }
        return self.state

    def chose_card(self, index):
        # This method compares the cards and ells if you win or lost
        if index not in [0,1]:
            print(f"wrong index {index}")
            sys.exit()
        if index==0 and self.state['card_1'] > self.state['card_2']:
            return 1  # wins
        elif index==1 and self.state['card_2'] > self.state['card_1']:
            return 1  # wins
        else:
            return -1 # lost



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class Player:
    def __init__(self, learning_rate, memory_size, exploration_rate=0.1):
        self.nn = MLP(input_dim=2, output_dim=2)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.memory = []
        self.memory_size = memory_size
        self.exploration_rate = exploration_rate

    def update_exploration_rate(self, win_rate):
        self.exploration_rate = max(0.01, (1.0 - win_rate)**2)

    def get_exploration_rate(self):
        return self.exploration_rate

    def choose_action(self, game_state, isTraing = False):
        state_tensor = torch.tensor([game_state['card_1'], game_state['card_2']], dtype=torch.float32)
        action_values = self.nn(state_tensor)
        
        # Add some exploration
        if isTraing and np.random.rand() < self.exploration_rate:
            action_index = np.random.randint(2)
        else:
            action_index = torch.argmax(action_values).item()

        return action_index

    def learn(self):
        if len(self.memory) < self.memory_size:
            return

        #print("learn")
        # Sample a batch of experiences from memory
        states, actions, rewards = zip(*self.memory)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        self.optimizer.zero_grad()

        # Compute Q(s,a)
        q_values = self.nn(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute the target value
        targets = rewards

        # Compute the loss
        loss = self.loss_fn(q_values, targets)

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        self.optimizer.step()
        #erase memory
        self.memory = []

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))
        if len(self.memory) > self.memory_size:
            print("errasing a training")
            sys.exit()


def train(player, game):
    game.reset()
    player_actions = defaultdict(int)
    number_of_games = 0
    while not game.last():
        number_of_games+=1
        game.next()
        action_index = player.choose_action(game.get_state(), True)
        reward = game.chose_card(action_index)
        player_actions[action_index]+=1
        state = game.get_state() # get the state after the action
        player.remember((state['card_1'], state['card_2']), action_index, reward) # remember the state, action and reward
        player.learn()
    
    print(f"number_of_games: {number_of_games}")

def test(player, game, num_games):
    player_wins = 0
    player_chose_card_1 = 0
    player_chose_card_2 = 0
    game_lost = []

    for i in range(num_games):
        #print(f"playing game {i}")
        game.reset()
        state = game.get_state()
        action_index = player.choose_action(state)
        reward = game.chose_card(action_index)

        if reward == 1:
            player_wins+=1
        else:
            game_lost.append((action_index, state))
        if action_index==1:
            player_chose_card_1+=1
        else:
            player_chose_card_2+=1

    return player_wins

# In your main function:
def main():
    # Instantiate the game
    int_max_to_chose = 1000
    game = CardGame(int_max_to_chose)
    print(f"learning to chose between number in [1,{int_max_to_chose}]")
    win_percent = 0
    # Instantiate the players
    player = Player(learning_rate=0.01, memory_size = 100)
    increment_training_games = 1000
    total_training = 0
    test_games = 1000
    while win_percent!=100:
        # Train the players
        
        train(player, game)
        total_training+=increment_training_games
        wins = test(player, game, test_games)
        win_percent = 100.0 * wins / (1.0*test_games)
        # Update exploration rate based on win percent
        player.update_exploration_rate(win_percent/100) 
        print(f"after {total_training} trainig we got {win_percent}% wins, exploration rate {player.get_exploration_rate()}")


if __name__ == "__main__":
    main()



