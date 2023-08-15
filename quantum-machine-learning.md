# Quantum computing will change machine learning

The findings of the South Korean physics researchers have been verified, and the significant physics discovery of our era that many of us thought had occured, has not happened yet. But the potential for superconductors that function at ambient pressure and room temperatures wil fundamentally change the qubit's role in quantum computing. This change will aid in scalability, although higher temperatures can still affect quantum inaccuracies. Nevertheless, room temperature superconductors represent a crucial element of the quantum conundrum, simplifying the creation of hybrid systems capable of alternating between classical and quantum computation tasks.

In the future, it might be feasible that quantum hardware and software will collaborate seamlessly with their classical counterparts, concentrating on calculations tailor-made for optimization challenges. Within the realms of machine learning and deep learning, this synergy could transform conventional AI paradigms.

Many quantum algorithms that have achieved exponential accelerations have done so through clever approaches applied to problems with explicit objectives, confirmable outcomes, and well thought out strategies toward a solution. In contrast to traditional machine learning methods, quantum algorithms will harness the unique characteristics of superposition, entanglement, and interference to discover alternative solutions where conventional computers falter. To understand how quantum technology could be applied to something like neural networks, we need to take a pause and start at square one.
<br/>

# What are qubits and gates?

In order to make this approachable, I will explain these concepts using an analogy.

<br/>

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/c2e04061-6988-4aee-980f-507a49086a6e" width="400" height="270">
</p>

<br/>

Imagine the bits (a binary digit and is the smallest unit of data that a computer can process and store) of a classical computer as a simple light switch in your home; they are either in the OFF (0) or ON (1) position. 

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/8377b945-903e-477c-86d2-535a847d9f78" width="400" height="270">
</p>

In the world of regular computing, this binary system forms the foundation of all operations.

Now, picture a qubit as a dimmer switch instead of a simple light switch. The dimmer switch not only has an ON and OFF position but also various stages in between, allowing for a wide range of light levels. This is a lot like how qubits work. They can represent not just 0 or 1 but also a superposition of both 0 and 1 simultaneously, or a range of values between 0 and 1. 

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/8fac1bdb-8283-4f36-97e9-7e3d4f745d14" width="300" height="370">
</p>


This ability to be in multiple states at once allows quantum computers to process a vast amount of information simultaneously, leading to potentially enormous computational power.

To understand quantum gates, imagine a toy train set. Classical gates (boolean logical operators in classical programming like AND, OR, NOT) in regular computing are akin to the set's basic pieces, which allow the train (data) to move in specific paths, join tracks, or reverse direction. They perform specific logical operations on the bits.
<br/>

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/636a8653-3d71-4dde-a2a7-d9321401c8da">
</p>

Quantum gates are like magical pieces in this train set that can rotate the tracks, twist them into three-dimensional shapes, or even create tunnels that the train can simultaneously travel through and not travel through. 

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/af2e156a-7e31-4da5-8285-180f752b261a">
</p>

These "magic" pieces perform specific operations on qubits, manipulating them in complex ways, which include superposition and entanglement (where qubits become interconnected and the state of one affects the others).

Now, take the concept of deep learning, a form of artificial intelligence where classical computers imitate the way our brains think, forming networks to "learn" from data. Picture it like a very intricate train set with many tracks, switches, and stations representing the complex relationships in the data.

In the context of quantum deep learning, replace this classical train set with our magical quantum train set, equipped with dimmer switches (qubits) and magical pieces (quantum gates). This allows for an extraordinarily sophisticated network that can explore multiple pathways simultaneously, make connections more quickly, and potentially unlock solutions that classical systems might never find.
<br/>

# Limitations

With the limitations of near-term quantum hardware potentially being a thing of the past soon as physics research advances, the maturity of classical algorithms is another matter that quantum algorithms will need to address. 

Let's walk through an example. I have the make moons dataset and BLANK BLANK BLANK. We can illustrate how effective classical machine learning algorithms are, how much better neural networks are, and how far quantum neural networks need to go before we can begin using the technology in a meaningful way.
<br/>

# Understanding the limits

The make moons dataset is a great tool for when you want to experiment with or visualize algorithms that deal with complex, non-linear relationships. Imagine looking up at the night sky and seeing two crescent moons touching at their tips. 

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/253302ea-6128-4a13-b873-c7657583b774">
</p>

This toy dataset consists of two interleaving half circles, or "moons," hence the name. If you plot the dataset, it looks like two shapes resembling crescent moons.

This dataset is often used for binary classification problems, where the objective is to categorize data points into one of two classes or groups. Because the two "moons" are intertwined, this dataset is particularly useful for testing algorithms that can handle non-linear boundaries between classes.


# Logistic regression

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/b300f0ba-6a59-49c9-948f-c14eb6f03d4a" width="400" height="300">
</p>

```python
data3, target3 = make_moons(n_samples = 70000, noise = 0.1)

# Split into training and test sets
train_data3, test_data3, train_target3, test_target3 = train_test_split(data3, target3, test_size = 0.2, random_state = 42)

# Normalization
scaler = StandardScaler()
train_data3 = scaler.fit_transform(train_data3)
test_data3 = scaler.transform(test_data3)

# Create and fit logistic regression model
model = LogisticRegression()
model.fit(train_data3, train_target3)

# Generate a grid over the feature space
logistic_x_min, logistic_x_max = test_data3[:, 0].min() - 1, test_data3[:, 0].max() + 1
logistic_y_min, logistic_y_max = test_data3[:, 1].min() - 1, test_data3[:, 1].max() + 1
logistic_xx, logistic_yy = np.meshgrid(np.linspace(logistic_x_min, logistic_x_max, 100),
                                       np.linspace(logistic_y_min, logistic_y_max, 100))

# Evaluate the model on the grid data
logistic_Z = model.predict_proba(np.c_[logistic_xx.ravel(), logistic_yy.ravel()])[:, 1]
logistic_Z = logistic_Z.reshape(logistic_xx.shape)

# Plot the decision boundary
plt.contourf(logistic_xx, logistic_yy, logistic_Z, levels = [0,0.5,1], cmap = 'coolwarm', alpha = 0.3)
plt.scatter(test_data3[:, 0], test_data3[:, 1], c = test_target3, cmap = 'coolwarm', edgecolors = 'k', marker = 'x')
plt.title('Logistic Regression Classifier')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

```python
Accuracy of logistic regression model: 88.47%
```

<br/>

# Multi-layer perceptron

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/f8d79691-e28c-49c4-a371-54e16c09ce66" width="440" height="330">
</p>

```python
data4, target4 = make_moons(n_samples = 70000, noise = 0.1)

# Split into training and test sets
train_data4, test_data4, train_target4, test_target4 = train_test_split(data4, target4, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
train_data4 = scaler.fit_transform(train_data4)
test_data4 = scaler.transform(test_data4)

# Convert to PyTorch tensors
train_data4 = torch.tensor(train_data4, dtype = torch.float32)
test_data4 = torch.tensor(test_data4, dtype = torch.float32)
train_target4 = torch.tensor(train_target4, dtype = torch.float32).view(-1, 1)
test_target4 = torch.tensor(test_target4, dtype = torch.float32).view(-1, 1)

# Define a neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 1)
        self.layer3 = nn.Linear(1, 5)
        self.layer4 = nn.Linear(5, 10)
        self.layer5 = nn.Linear(10, 20)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
    
    
# Neural network model    
model = NeuralNetwork()
# Loss function
loss_func = nn.BCELoss()

# Optimization
optimizer = optim.Adam(model.parameters(), lr = 0.1)
# Optimization Loop
for epoch in range(30):
    optimizer.zero_grad()
    predictions = model(train_data4)
    loss = loss_func(predictions, train_target4)
    loss.backward()
    optimizer.step()
    print('Epoch: {} | Loss: {}'.format(epoch, loss.item()))
    
# Plotting the training data
plt.scatter(train_data4[:, 0], train_data4[:, 1], c = train_target4[:, 0], cmap = 'coolwarm', edgecolors = 'k', alpha = 0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Generate a grid over the feature space
resolution = 100
x_range = np.linspace(train_data4[:, 0].min() - 1, train_data4[:, 0].max() + 1, resolution)
y_range = np.linspace(train_data4[:, 1].min() - 1, train_data4[:, 1].max() + 1, resolution)
grid_x, grid_y = np.meshgrid(x_range, y_range)
grid_data = torch.tensor(np.c_[grid_x.ravel(), grid_y.ravel()], dtype = torch.float32)

# Evaluate the model on the grid data
grid_predictions = model(grid_data)
grid_predictions = grid_predictions.detach().numpy().reshape(grid_x.shape)

# Plot the decision boundary 
plt.contourf(grid_x, grid_y, grid_predictions, levels = [0,0.5,1], cmap = 'coolwarm', alpha = 0.3)
plt.colorbar(label = 'Prediction Probability')

test_predictions_continuous = model(test_data4).detach().numpy()

plt.scatter(test_data4[:, 0], test_data4[:, 1], c = test_predictions_continuous, cmap = 'coolwarm', edgecolors = 'k', marker = 'x')
plt.title('Neural Network Classifier')
plt.show()
```

```python 
Accuracy of neural network model: 94.99%
```

<br/>

# Quantum multi-layer perceptron

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/3a99e2aa-43d3-42aa-8e54-4f9cd4782e03" width="400" height="300">
</p>

```python
# Load dataset and select the first two features
data, target = make_moons(n_samples = 70000, noise = 0.1)

# Normalization
data = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)

# Split into training and test sets
train_data, test_data = data[:160], data[160:]
train_target, test_target = target[:160], target[160:]

# Define a quantum device
dev = qml.device('default.qubit', wires = 2)

# Define a quantum layer
def layer(W):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires = 0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires = 1)
    qml.CNOT(wires = [0, 1])

# Quantum node
@qml.qnode(dev, interface = 'torch')
def quantum_net(features, weights):
    padded_features = np.pad(features, (0, 2 - len(features)), constant_values = 0)
    qml.templates.AngleEmbedding(padded_features, wires = [0, 1])
    for W in weights:
        # Increases the number of repetitions of the layer
        for _ in range(8): 
            layer(W)
        #layer(W)
    return qml.expval(qml.PauliZ(0))

# Hybrid quantum-classical model
def hybrid_model(x, weights):
    pre_sigmoid_predictions = torch.tensor([quantum_net(x_, weights) for x_ in x], requires_grad = True)
    return F.sigmoid(pre_sigmoid_predictions)

# Loss function
loss_func = torch.nn.BCEWithLogitsLoss()

# Define weights
weights = torch.tensor(np.random.random(size = (2, 2, 3)), requires_grad = True)

# Optimization Loop
optimizer = torch.optim.AdamW([weights], lr = 0.1)
for epoch in range(10):
    optimizer.zero_grad()
    predictions = hybrid_model(train_data, weights).float()
    loss = loss_func(predictions, torch.tensor(train_target, dtype = torch.float))
    loss.backward()
    optimizer.step()
    print('Epoch: {} | Loss: {}'.format(epoch, loss.item()))

# Create a mesh grid
x_min, x_max = test_data[:, 0].min() - 1, test_data[:, 0].max() + 1
y_min, y_max = test_data[:, 1].min() - 1, test_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Apply the model to each point on the grid
Z = hybrid_model(np.c_[xx.ravel(), yy.ravel()], weights)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z.detach().numpy(), levels = [0,0.5,1], cmap = 'coolwarm', alpha = 0.3)
plt.scatter(test_data[:, 0], test_data[:, 1], c = test_target, cmap = 'coolwarm', edgecolors = 'k', marker = 'x')
plt.title('Quantum Neural Network Classifier')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

```python
Accuracy of neural quantum model: 45.69%
```
