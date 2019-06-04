import numpy as np
import utils
import os
from collections import deque

class BaseModel(object):
    def __init__(self, gridSize, numActions=4, imgSize=28, step=7, historySize=1, epsilon=0.1, seed=42):

        np.random.seed(seed)
        self.gridSize = gridSize
        self.imgSize = imgSize
        self.step = step
        self.historySize = historySize
        self.numActions = numActions

        self.history = deque(maxlen=self.historySize)

        self.numStates = self.gridSize**2
        for i in range(historySize):
            self.numStates += self.numActions ** (i + 1) * self.gridSize**2

        self.policy = np.zeros((self.numStates, numActions))
        self.stateAction = np.full((self.numStates, self.numActions), np.nan)
        self.stateActionSequence = None
        self.stateActionCount = np.ones_like(self.stateAction, dtype=np.int)
        self.epsilon = epsilon

        self.initStateAction()
        self.initPolicy()

    def initStateAction(self):
        for i in range(self.numStates):
            offset = utils.getOffset(i % self.gridSize**2, self.gridSize, self.step)
            possibleActions = utils.getPossibleActions(offset, self.imgSize, self.step)
            self.stateAction[i, possibleActions] = 0

    def initPolicy(self):
        for i in range(self.numStates):
            mask = ~np.isnan(self.stateAction[i])
            self.policy[i, mask] = 1. / np.count_nonzero(mask)

    def createSequence(self, startState, length, random=True):
        '''
        Creates state-action sequence, starting from startState and following current policy
        '''
        sequence = []
        currentState = startState
        stateActionSequence = []

        for i in range(length):
            if random:
                action = np.random.choice([0, 1, 2, 3], p=self.policy[currentState])
            else:
                action = np.argmax(self.policy[currentState])
            self.history.append(action)
            stateActionSequence.append((currentState, action))
            sequence.append(action)
            currentState = utils.updateState(currentState, action, self.history, self.gridSize, self.step)

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

    def CenterSequence(self, length):
        start = self.gridSize // 2 * self.gridSize + self.gridSize // 2
        return utils.getOffset(start, self.gridSize, self.step), self.createSequence(start, length, random=True)

    def getStart(self):
        state = np.random.randint(0, self.gridSize ** 2)
        action = np.random.choice(self.stateAction[state, ~np.isnan(self.stateAction[state])])
        action, = np.where(self.stateAction[state, :] == action)
        return state, action[0]

    def BestSequence(self, length):
        state = self.gridSize // 2 * self.gridSize + self.gridSize // 2
        sequence = self.createSequence(state, length, random=False)
        return utils.getOffset(state, self.gridSize, self.step), sequence

    def RandomStartSequence(self, length):
        assert length > 0
        start, action = self.getStart()
        self.history.append(action)
        nextState = utils.updateState(start, action, self.history, self.gridSize, self.step)
        sequence = [action]
        sequence.extend(self.createSequence(nextState, length - 1))
        self.stateActionSequence.append((start, action))
        return utils.getOffset(start % self.gridSize**2, self.gridSize, self.step), sequence

    def save(self, dirPath):
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        count = len([name for name in os.listdir(dirPath)]) // 3
        policyPath = os.path.join(dirPath, 'policy_%02d.txt' % count)
        stateActPath = os.path.join(dirPath, 'stateAction_%02d.txt' % count)
        countPath = os.path.join(dirPath, 'stateActionCount_%02d.txt' % count)

        np.savetxt(policyPath, self.policy)
        np.savetxt(stateActPath, self.stateAction)
        np.savetxt(countPath, self.stateActionCount)

    def load(self, dirPath, version='latest'):
        if version == 'latest':
            count = len([name for name in os.listdir(dirPath) if name]) // 3
            policyPath = os.path.join(dirPath, 'policy_%02d.txt' % (count - 1))
            stateActPath = os.path.join(dirPath, 'stateAction_%02d.txt' % (count - 1))
            stateActCountPath = os.path.join(dirPath, 'stateActionCount_%02d.txt' % (count - 1))
        elif isinstance(version, (int, long)):
            policyPath = os.path.join(dirPath, 'policy_%02d.txt' % version)
            stateActPath = os.path.join(dirPath, 'stateAction_%02d.txt' % version)
            stateActCountPath = os.path.join(dirPath, 'stateActionCount_%02d.txt' % version)
        else:
            raise AttributeError('Unknown version')

        if os.path.exists(policyPath):
            self.policy = np.loadtxt(policyPath)
        else:
            print 'Failed to load from %s' % policyPath

        if os.path.exists(stateActPath):
            self.stateAction = np.loadtxt(stateActPath)
        else:
            print 'Failed to load from %s' % stateActPath

        if os.path.exists(stateActCountPath):
            self.stateActionCount = np.loadtxt(stateActCountPath)
        else:
            print 'Failed to load from %s' % stateActCountPath


class CategoryModel(object):
    def __init__(self, numCategories, numActions=4, epsilon=0.1, seed=42, step=7, imgSize=28, ignoreCategory=False):
        np.random.seed(seed)
        self.imgSize = imgSize
        self.step = step
        self.gridSize = imgSize // step
        if ignoreCategory:
            self.numCategories = 1
        else:
            self.numCategories = numCategories
        self.numStates = self.gridSize ** 2
        self.numActions = numActions
        self.ignoreCategory = ignoreCategory
        self.stateAction = np.full((numCategories, self.numStates, self.numActions), np.nan)
        self.policy = np.zeros_like(self.stateAction)
        self.epsilon = epsilon
        self.stateActionSequence = None
        self.stateActionCount = np.ones_like(self.stateAction, dtype=np.int)
        self.initStateAction()
        self.initPolicy()

    def initStateAction(self):
        for category in range(self.numCategories):
            for i in range(self.numStates):
                offset = utils.getOffset(i % self.gridSize**2, self.gridSize, self.step)
                possibleActions = utils.getPossibleActions(offset, self.imgSize, self.step)
                self.stateAction[category, i, possibleActions] = 0

    def initPolicy(self):
        for category in range(self.numCategories):
            for i in range(self.numStates):
                mask = ~np.isnan(self.stateAction[category, i])
                self.policy[category, i, mask] = 1. / np.count_nonzero(mask)

    def createSequence(self, category, startPosition, length, random=True, store=True):
        '''
        Creates state-action sequence, starting from startState and following current policy
        '''
        sequence = []
        startState = utils.getIndex(startPosition, self.gridSize, self.step)
        currentState = startState
        stateActionSequence = []

        if self.ignoreCategory:
            category = 0

        for i in range(length):
            if random:
                action = np.random.choice([0, 1, 2, 3], p=self.policy[category, currentState])
            else:
                action = np.argmax(self.policy[category, currentState])
            stateActionSequence.append((currentState, action))
            sequence.append(action)
            currentState = utils.updateState(currentState, action, [], self.gridSize, self.step)
        if store:
            self.stateActionSequence = stateActionSequence[::-1]
            #self.stateActionSequence = self.stateActionSequence[1:] # last state doesn't have an action
        return sequence

    def update(self, category, reward):
        if self.ignoreCategory:
            category = 0
        self.stateActionSequence = self.stateActionSequence[1:]
        for i, (state, action) in enumerate(self.stateActionSequence):
            if (state, action) not in self.stateActionSequence[i + 1:]:
                self.stateAction[category, state, action] += \
                    1. / self.stateActionCount[category, state, action] * \
                    (reward - self.stateAction[category, state, action])
                self.stateActionCount[category, state, action] += 1
                action = np.nanargmax(self.stateAction[category, state])
                for act in range(self.numActions):
                    if self.policy[category, state, act] > 0:
                        if act == action:
                            self.policy[category, state, act] = \
                                1 - self.epsilon + self.epsilon / np.count_nonzero(self.policy[category, state])
                        else:
                            self.policy[category, state, act] = \
                                self.epsilon / np.count_nonzero(self.policy[category, state])

    def BestSequence(self, category, startPosition, length):
        if self.ignoreCategory:
            category = 0
        sequence = self.createSequence(category, startPosition, length, random=False)
        return sequence

    def getNextAction(self, categories, probs, position):
        possibleActions = np.zeros_like(categories, dtype=np.int)
        for i, category in enumerate(categories):
            action = self.createSequence(category, position, 1, random=False, store=False)[0]
            possibleActions[i] = action

        probs = np.exp(probs)/sum(np.exp(probs))

        #return np.random.choice(possibleActions, p=probs)
        counts = np.bincount(possibleActions)
        return np.argmax(counts)

    def save(self, dirPath):
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        count = len([name for name in os.listdir(dirPath)]) // 3
        policyPath = os.path.join(dirPath, 'policy_%02d' % count)
        stateActPath = os.path.join(dirPath, 'stateAction_%02d' % count)
        countPath = os.path.join(dirPath, 'stateActionCount_%02d' % count)

        np.save(policyPath, self.policy)
        np.save(stateActPath, self.stateAction)
        np.save(countPath, self.stateActionCount)

    def load(self, dirPath, version='latest'):
        if version == 'latest':
            count = len([name for name in os.listdir(dirPath) if name]) // 3
            policyPath = os.path.join(dirPath, 'policy_%02d.txt' % (count - 1))
            stateActPath = os.path.join(dirPath, 'stateAction_%02d.txt' % (count - 1))
            stateActCountPath = os.path.join(dirPath, 'stateActionCount_%02d.txt' % (count - 1))
        elif isinstance(version, (int, long)):
            policyPath = os.path.join(dirPath, 'policy_%02d.txt' % version)
            stateActPath = os.path.join(dirPath, 'stateAction_%02d.txt' % version)
            stateActCountPath = os.path.join(dirPath, 'stateActionCount_%02d.txt' % version)
        else:
            raise AttributeError('Unknown version')

        if os.path.exists(policyPath):
            self.policy = np.loadtxt(policyPath)
        else:
            print 'Failed to load from %s' % policyPath

        if os.path.exists(stateActPath):
            self.stateAction = np.loadtxt(stateActPath)
        else:
            print 'Failed to load from %s' % stateActPath

        if os.path.exists(stateActCountPath):
            self.stateActionCount = np.loadtxt(stateActCountPath)
        else:
            print 'Failed to load from %s' % stateActCountPath



if __name__ == '__main__':
    model = CategoryModel(10, seed=5, ignoreCategory=True)
    print model.createSequence(5, [7, 7], 5)
    model.update(5, 100)
    print model.createSequence(5, [7, 7], 5)
    print model.createSequence(3, [7, 7], 5)