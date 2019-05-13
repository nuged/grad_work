import numpy as np
import utils
import os


class BaseModel(object):
    def __init__(self, numActions, epsilon=0.1, imgSize=28, seed=42, step=7):
        np.random.seed(seed)
        gridSize = imgSize // step
        self.gridSize = gridSize
        self.numStates = gridSize * gridSize * (numActions + 1)
        self.numActions = numActions
        self.policy = np.zeros((self.numStates, numActions))
        self.stateAction = np.full((self.numStates, self.numActions), np.nan)
        self.stateActionSequence = None
        self.stateActionCount = np.ones_like(self.stateAction, dtype=np.int)
        self.epsilon = epsilon
        self.initStateAction()
        self.initPolicy()

    def initStateAction(self):
        for i in range(self.numStates):
            offset = utils.getOffset(i % self.gridSize**2)
            possibleActions = utils.getPossibleActions(offset)
            self.stateAction[i, possibleActions] = 0

    def initPolicy(self):
        for i in range(self.numStates):
            mask = ~np.isnan(self.stateAction[i])
            self.policy[i, mask] = 1. / np.count_nonzero(mask)

    def createActSequence(self, startState, length, random=True):
        sequence = []
        currentState = startState
        stateActionSequence = []
        for i in range(length):
            if random:
                action = np.random.choice([0, 1, 2, 3], p=self.policy[currentState])
            else:
                action = np.argmax(self.policy[currentState])
            stateActionSequence.append((currentState, action))
            sequence.append(action)
            currentState = utils.updateState(currentState, action)
        self.stateActionSequence = stateActionSequence[::-1]
        return sequence

    def update(self, reward):
        for i, (state, action) in enumerate(self.stateActionSequence):
            if (state, action) not in self.stateActionSequence[i + 1:]:
                self.stateAction[state, action] += \
                    1. / self.stateActionCount[state, action] * (reward - self.stateAction[state, action])
                self.stateActionCount[state, action] += 1
                action = np.nanargmax(self.stateAction[state])
                for act in range(self.numActions):
                    if self.policy[state, act] > 0:
                        if act == action:
                            self.policy[state, act] = \
                                1 - self.epsilon + self.epsilon / np.count_nonzero(self.policy[state])
                        else:
                            self.policy[state, act] = \
                                self.epsilon / np.count_nonzero(self.policy[state])

    def getStart(self):
        state = np.random.randint(0, self.gridSize ** 2)
        action = np.random.choice(self.stateAction[state, ~np.isnan(self.stateAction[state])])
        action, = np.where(self.stateAction[state, :] == action)
        return state, action[0]

    def BestSequence(self, length):
        state = np.nanargmax(self.stateAction[:self.gridSize**2]) // self.numActions
        sequence = self.createActSequence(state, length, random=False)
        return utils.getOffset(state), sequence

    def RandomStartSequence(self, length):
        assert length > 0
        start, action = self.getStart()
        nextState = utils.updateState(start, action)
        sequence = [action]
        sequence.extend(self.createActSequence(nextState, length - 1))
        self.stateActionSequence.append((start, action))
        return utils.getOffset(start % self.gridSize**2), sequence

    def save(self, dirPath):
        count = len([name for name in os.listdir(dirPath)]) // 3
        policyPath = os.path.join(dirPath, 'policy_%02d.npy' % count)
        stateActPath = os.path.join(dirPath, 'stateAction_%02d.npy' % count)
        countPath = os.path.join(dirPath, 'stateActionCount_%02d.npy' % count)
        np.save(policyPath, self.policy)
        np.save(stateActPath, self.stateAction)
        np.save(countPath, self.stateActionCount)

    def load(self, dirPath, version='latest'):
        if version == 'latest':
            count = len([name for name in os.listdir('.') if name]) // 3
            policyPath = os.path.join(dirPath, 'policy_%02d.npy' % count)
            stateActPath = os.path.join(dirPath, 'stateAction_%02d.npy' % count)
            stateActCountPath = os.path.join(dirPath, 'stateActionCount_%02d.npy' % count)
        elif isinstance(version, (int, long)):
            policyPath = os.path.join(dirPath, 'policy_%02d.npy' % version)
            stateActPath = os.path.join(dirPath, 'stateAction_%02d.npy' % version)
            stateActCountPath = os.path.join(dirPath, 'stateActionCount_%02d.npy' % version)
        else:
            raise AttributeError('Unknown version')

        if os.path.exists(policyPath):
            self.policy = np.load(policyPath)
        else:
            print 'Failed to load from %s' % policyPath

        if os.path.exists(stateActPath):
            self.stateAction = np.load(stateActPath)
        else:
            print 'Failed to load from %s' % stateActPath

        if os.path.exists(stateActCountPath):
            self.stateActionCount = np.load(stateActCountPath)
        else:
            print 'Failed to load from %s' % stateActCountPath

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
    model = BaseModel(4, seed=90)
    model.load('models/epsilon-greedy', version=13)
    start, acts = model.BestSequence(6)
    acts = map(int, acts)
    print type(start[0])