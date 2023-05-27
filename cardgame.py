import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from collections import defaultdict

class CardGame:
    def __init__(self):
        self.reset()

    def get_state(self):
        return self.state

    def reset(self):
        # At the start of each game, randomly assign each player a card from 1 to 100.
        player_cards = np.random.choice(range(1, 7), size=2, replace=False)
        self.state = {
            'player1_card': player_cards[0],
            'player2_card': player_cards[1],
        }
        return self.state

    def step(self, player, action):
        # This method handles a player's action and updates the game state.
        done = False
        reward = 0

        if player == 1:
            if action == 'propose_exchange':
                pass
            elif action == 'show_card':
                reward = self._compare_cards()
                done = True
            else:
                print(f"invalid action for player 1: {action}")
                sys.exit()
        elif player == 2:
            if action == 'accept_exchange':
                self._exchange_cards()
                reward = self._compare_cards() * -1  # Flip the reward for player 2's perspective
                done = True
            elif action == 'show_card':
                reward = self._compare_cards() * -1  # Flip the reward for player 2's perspective
                done = True
            else:
                print(f"invalid action for player 2: {action}")
                sys.exit()

        return self.state, reward, done


    def _compare_cards(self):
        # This method compares the players' cards and determines the winner.
        if self.state['player1_card'] > 2*self.state['player2_card']:
            return 1  # Player 1 wins
        else:
            return -1  # Player 2 wins


    def _exchange_cards(self):
        # This method exchanges the players' cards.
        self.state['player1_card'], self.state['player2_card'] = self.state['player2_card'], self.state['player1_card']



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
    def __init__(self, playerID, learning_rate, discount_factor):
        self.nn = MLP(input_dim=2, output_dim=2)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.discount_factor = discount_factor
        self.playerID = playerID


    def choose_action(self, game_state):
        state_tensor = torch.tensor([game_state['player1_card'], game_state['player2_card']], dtype=torch.float32)
        action_values = self.nn(state_tensor)
        action_index = torch.argmax(action_values).item()
        if self.playerID == 1:
            action = ['propose_exchange', 'show_card'][action_index]
        else:
            action = ['accept_exchange', 'show_card'][action_index]
        return action_index, action

    def learn(self, old_state, action_index, reward):
        # Convert the state dictionaries into tensors
        old_state_tensor = torch.tensor([old_state['player1_card'], old_state['player2_card']], dtype=torch.float32)
        
        reward = torch.tensor(reward, dtype=torch.float)  # convert reward to float
        
        self.optimizer.zero_grad()

        # Compute Q(s,a)
        q_values_old = self.nn(old_state_tensor)
        q_value_old = q_values_old[action_index].unsqueeze(0)  # Add an extra dimension to the predicted Q-value

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


def train(player1, player2, game, num_games):
    player_1_actions = defaultdict(int)
    player_2_actions = defaultdict(int)
    for i in range(num_games):
        print(f"training game {i}")
        game.reset()
        done = False
        while not done:
            # Player 1's turn
            state_1 = game.get_state()
            action_index_1, action_1 = player1.choose_action(state_1)
            player_1_actions[action_1]+=1
            _, reward, done = game.step(1, action_1)
            #player1.learn(state_1, action_index_1, reward)

            if done:
                print(f"player 1 {reward} {done} {action_1}")
                player1.learn(state_1, action_index_1, -1)
                continue

            # Player 2's turn
            state_2 = game.get_state()
            action_index_2, action_2 = player2.choose_action(state_2)
            _, reward, done = game.step(2, action_2)
            print(f"player 1 {reward} {done} {action_1}")
            print(f"player 2 {reward} {done} {action_2}")
            player_2_actions[action_2]+=1
            player1.learn(state_1, action_index_1, 1)
            player2.learn(state_2, action_index_2, reward)

    print('player_1_actions ' , dict(player_1_actions))
    print('player_2_actions ' , dict(player_2_actions))


def test(player1, player2, game, num_games):
    player1_wins = 0
    player2_wins = 0
    player1_exchange_cards = []
    player1_no_exchange_cards = []
    player2_accepted_cards = []
    player2_rejected_cards = []

    for i in range(num_games):
        print(f"playing game {i}")
        game.reset()
        done = False
        while not done:
            state = game.get_state()
            _, action = player1.choose_action(state)

            if action == 'propose_exchange':
                player1_exchange_cards.append((state['player1_card'], state['player2_card']))
            elif action == 'show_card':
                print(f"{action} player 1")
                player1_no_exchange_cards.append((state['player1_card'], state['player2_card']))
            else:
                print("invalid action for player 2: {action}")
                sys.exit()

            _, reward, done = game.step(1, action)
            if done:
                if reward > 0:
                    player1_wins += 1
                else:
                    player2_wins += 1
                continue

            state = game.get_state()
            _, action = player2.choose_action(state)

            if action == 'accept_exchange':
                player2_accepted_cards.append((state['player1_card'], state['player2_card']))
            elif action == 'show_card':
                player2_rejected_cards.append((state['player1_card'], state['player2_card']))
            else:
                print("invalid action for player 2: {action}")
                sys.exit()

            _, reward, done = game.step(2, action)
            if done:
                if reward > 0:
                    player2_wins += 1
                else:
                    player1_wins += 1

    print('Player 1 wins:', player1_wins)
    print('Player 2 wins:', player2_wins)
    print('Player 1 proposed exchange with cards:', player1_exchange_cards)
    print('Player 1 did not propose exchange with cards:', player1_no_exchange_cards)
    print('Player 2 accepted exchange with cards:', player2_accepted_cards)
    print('Player 2 rejected exchange with cards:', player2_rejected_cards)

# In your main function:
def main():
    # Instantiate the game
    game = CardGame()

    # Instantiate the players
    player1 = Player(1, learning_rate=0.01, discount_factor=0.99)
    player2 = Player(2, learning_rate=0.01, discount_factor=0.99)

    # Train the players
    num_games = 10000
    train(player1, player2, game, num_games)

    # Test the players
    num_games = 20
    test(player1, player2, game, num_games)

if __name__ == "__main__":
    main()



