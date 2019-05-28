from sklearn.neural_network import MLPClassifier
import numpy as np

model = MLPClassifier((2048, 256), activation='relu', solver='adam', learning_rate='adaptive',
                      learning_rate_init=1e-2, warm_start=True, batch_size=16)
np.random.seed(5)
data = np.random.randint(0, 2, (100, 4096*4))
answer = np.random.randint(0, 10, (100,))
model.fit(data, answer)

data = np.random.randint(0, 2, (100, 4096*4))
answer = np.random.randint(0, 10, (100,))
model.fit(data, answer)

np.random.seed(5)
data = np.random.randint(0, 2, (10, 4096*4))
answer = np.random.randint(0, 10, (10,))
print model.predict(data), answer