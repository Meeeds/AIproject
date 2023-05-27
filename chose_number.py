import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from collections import defaultdict

class CardGame:
    def __init__(self, number_of_cards):
        self.number_of_cards = number_of_cards
        self.reset()
        

    def get_state(self):
        return self.state

    def reset(self):
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
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class Player:
    def __init__(self, learning_rate, discount_factor):
        self.nn = MLP(input_dim=2, output_dim=2)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.discount_factor = discount_factor

    def choose_action(self, game_state):
        state_tensor = torch.tensor([game_state['card_1'], game_state['card_2']], dtype=torch.float32)
        action_values = self.nn(state_tensor)
        action_index = torch.argmax(action_values).item()
        return action_index

    def learn(self, old_state, action_index, reward):
        # Convert the state dictionaries into tensors
        old_state_tensor = torch.tensor([old_state['card_1'], old_state['card_2']], dtype=torch.float32)
        
        reward = torch.tensor(reward, dtype=torch.float)  # convert reward to float
        
        self.optimizer.zero_grad()

        # Compute Q(s,a)
        q_values_old = self.nn(old_state_tensor)
        q_value_old = q_values_old[action_index].unsqueeze(0) # Add an extra dimension to the predicted Q-value

        # Compute the target value
        target = reward

        # Compute the loss
        loss = self.loss_fn(q_value_old, torch.tensor([target]))

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        self.optimizer.step()
        #print(f'Loss: {loss.item()}', end='')
        #for name, param in self.nn.named_parameters():
            #if param.requires_grad:
                #print(name, param.data)


def train(player, game, num_games):
    player_actions = defaultdict(int)
    for i in range(num_games):
        #print(f"training game {i}")
        game.reset()
        action_index = player.choose_action(game.get_state())
        reward = game.chose_card(action_index)
        #print(f"player {reward} {action_index}")
        player_actions[action_index]+=1
        player.learn(game.get_state(), action_index, reward)

    #print('player_actions ' , dict(player_actions))
    

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

    #print('Player wins:', player_wins)
    #print('chose first card', player_chose_card_1)
    #print('chose second card', player_chose_card_2)
    #print('game_lost', game_lost)

    return player_wins

# In your main function:
def main():
    # Instantiate the game
    int_max_to_chose = 100
    game = CardGame(int_max_to_chose)
    print(f"learning to chose between number in [1,{int_max_to_chose}]")
    win_percent = 0
    # Instantiate the players
    player = Player(learning_rate=0.01, discount_factor=0.99)
    increment_training_games = 1000
    total_training = 0
    test_games = 1000
    while win_percent!=100:
        # Train the players
        
        train(player, game, increment_training_games)
        total_training+=increment_training_games
        wins = test(player, game, test_games)
        win_percent = 100.0 * wins / (1.0*test_games)
        print(f"after {total_training} trainig we got {win_percent}% wins")


if __name__ == "__main__":
    main()



