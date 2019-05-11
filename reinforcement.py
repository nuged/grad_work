import numpy as np
import utils
import os

class BaseModel(object):
    def __init__(self, numActions, imgSize=28, seed=42):
        np.random.seed(seed)
        self.numStates = imgSize * imgSize
        self.numActions = numActions
        self.policy = np.zeros(self.numStates, dtype=np.int)
        self.stateAction = np.full((self.numStates, self.numActions), np.nan)
        self.stateActionSequence = None
        self.stateActionCount = np.ones_like(self.stateAction, dtype=np.int)

    def initStateAction(self):
        for i in range(self.numStates):
            offset = utils.getOffset(i)
            possibleActions = utils.getPossibleActions(offset)
            self.stateAction[i, possibleActions] = np.random.uniform(-100, 0, possibleActions.shape[0])

    def initPolicy(self):
        for i in range(self.numStates):
            action = np.nanargmax(self.stateAction[i])
            self.policy[i] = action

    def createActSequence(self, startState, length):
        sequence = []
        currentState = startState
        stateActionSequence = []
        for i in range(length):
            action = self.policy[currentState]
            stateActionSequence.append((currentState, action))
            sequence.append(action)
            currentState = utils.updateState(currentState, action)
        self.stateActionSequence = stateActionSequence[::-1]
        return sequence

    def update(self, reward):
        for i, (state, action) in enumerate(self.stateActionSequence):
            if (state, action) not in self.stateActionSequence[i + 1:]:
                self.stateAction[state, action] += 1. / self.stateActionCount[state, action] *\
                                                   (reward - self.stateAction[state, action])
                self.stateActionCount[state, action] += 1
                self.policy[state] = np.nanargmax(self.stateAction[state])

    def save(self, dirPath):
        count = len([name for name in os.listdir(dirPath)]) // 2
        policyPath = os.path.join(dirPath, 'policy_%02d.npy' % count)
        stateActPath = os.path.join(dirPath, 'stateAction_%02d.npy' % count)
        np.save(policyPath, self.policy)
        np.save(stateActPath, self.stateAction)

    def load(self, dirPath, version='latest'):
        if version == 'latest':
            count = len([name for name in os.listdir('.') if os.path.isfile(name)]) // 2
            policyPath = os.path.join(dirPath, 'policy_%2d.npy' % count)
            stateActPath = os.path.join(dirPath, 'stateAction_%2d.npy' % count)
        elif isinstance(version, (int, long)):
            policyPath = os.path.join(dirPath, 'policy_%2d.npy' % version)
            stateActPath = os.path.join(dirPath, 'stateAction_%2d.npy' % version)
        else:
            raise AttributeError('Unknown version')

        self.policy = np.load(policyPath)
        self.stateAction = np.load(stateActPath)

class ESModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super(ESModel, self).__init__(*args, **kwargs)
        self.initStateAction()
        self.initPolicy()

    def getStart(self):
        state = np.random.randint(0, self.numStates)
        action = np.random.choice(self.stateAction[state, ~np.isnan(self.stateAction[state])])
        action, = np.where(self.stateAction[state, :] == action)
        return state, action[0]

    def createESSequence(self, length):
        assert length > 0
        start, action = self.getStart()
        nextState = utils.updateState(start, action)
        sequence = [action]
        sequence.extend(self.createActSequence(nextState, length - 1))
        self.stateActionSequence.append((start, action))
        return utils.getOffset(start), sequence


if __name__ == '__main__':

    seed = np.random.randint(0, 256)
    model = ESModel(4, seed=seed)
    print seed
    print model.createESSequence(10)
    print model.stateActionSequence