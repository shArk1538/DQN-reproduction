### Install the OpenAI Gymnasium library
```
pip install gymnasium
```
### Run this project
```
python main.py
```
*you can set the hyperparameters by youself*
### Python files description
1. Make sure you install the pytorch GPU version to speed up training
2. **main.py:** create the game environment and start train loop
3. **Preprocessing.py:** detail of the official preprocess function
4. **ExpReplay.py:** experience replay part
5. **Network.py:** DQN sturcture
6. **DQN_Agent.py:** create the agent
7. **Train.py:** deep Q-learning with experience replay
