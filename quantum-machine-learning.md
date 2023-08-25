# Quantum computing will change everything

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

This dataset is often used for binary classification problems, where the objective is to categorize data points into one of two classes or groups. Because the two "moons" are intertwined, this dataset is particularly useful for testing algorithms that can handle non-linear boundaries between classes. Lets observe how logistic regression, a multi-layer perceptron and a hybrid quantum neural network classify the moon data:


## Logistic regression

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

## Multi-layer perceptron

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/f8d79691-e28c-49c4-a371-54e16c09ce66" width="440" height="330">
</p>

```python
data4, target4 = make_moons(n_samples = 70000, noise = 0.1)

# Split into training and test sets
train_data4, test_data4, train_target4, test_target4 = train_test_split(data4, target4, test_size = 0.2, random_state = 42)

# Normalization
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

## Quantum multi-layer perceptron

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/3a99e2aa-43d3-42aa-8e54-4f9cd4782e03" width="400" height="300">
</p>

```python
# Load dataset and select the first two features
data, target = make_moons(n_samples = 70000, noise = 0.1)

# Normalization
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split data into training and test sets
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state = 42)

# Define a quantum device
dev = qml.device('default.qubit', wires = 2)

# Define a quantum layer
def layer(W):
    qml.Rot(W[0], W[1], W[2], wires=0)
    qml.Rot(W[3], W[4], W[5], wires=1)
    qml.CNOT(wires=[0, 1])

# Quantum node
@qml.qnode(dev, interface='torch')
def quantum_net(features, *weights):
    padded_features = np.pad(features, (0, 2 - len(features)), constant_values=0)
    qml.templates.AngleEmbedding(padded_features, wires=[0, 1])
    for W in weights:
        layer(W)
    return qml.expval(qml.PauliZ(0))

# Hybrid quantum-classical model
def hybrid_model(x, weights):
    # Unpack the weight layers for passing to quantum_net
    weight_layers = [weights[i] for i in range(weights.shape[0])]
    pre_sigmoid_predictions = torch.tensor([quantum_net(x_, *weight_layers) for x_ in x], requires_grad=True)
    return torch.sigmoid(pre_sigmoid_predictions)

# Loss function
#loss_func = torch.nn.BCEWithLogitsLoss()
loss_func = torch.nn.BCELoss()

# Define weights for 5 layers
num_layers = 5
weights = torch.tensor(np.random.random(size=(num_layers, 6)), requires_grad=True)

# Optimization Loop
optimizer = torch.optim.AdamW([weights], lr = 0.001)
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
Accuracy of neural quantum model: 67.04%
```

<br>

Quantum neural networks are simply computational models based on quantum mechanics and they derive principles from classical deep learning. Assuming we're all aware of the complexities of logistic regression and neural networks, the hybrid quantum neural network's qubits are analogous to ANN neurons(or circuits) and are connected by wires that act as unitaries, which are simply our gates that we covered earlier, to apply operations to the qubits. 

Information is processed in HQNNs by all of the qubits in the network first being in a zero state. When information in passed to the input layer of the network, the qubits in this layers are passed to the next layer as tensors and are now in the hidden layers. The tensor is then passed to the output layer.

<p align="center">
<img src="https://github.com/christianThardy/Logistic-Regression/assets/29679899/b79ce7f4-3d3b-47a7-b099-f4ce31f3be05">
</p>

<br>

While trainable, HQNNs have a number of challenges that make them unreliable at the moment. 

## 1. Current quantum devices are noisy and error-prone

Even small quantum circuits on near-term quantum computers can be affected by noise, leading to inaccurate computations. Training a quantum neural network in an environment like this can be challenging because discerning whether a bad outcome is due to the noise or due to incorrect parameters of the model is hard to differentiate. The noise associated with longer computation makes deep quantum circuts unfeasible, so classical deep learning techniques suffer.  

## 2. Manipulating and understanding high-dimensional quantum states can be very challenging

Quantum gate design is nontrivial because they need to rotate in particular ways and this needs to be accounted for. Classical gradient-based optimization techniques very inefficient because in quantum optimization landscapes, regions known as "barren plateaus" make gradients nearly zero, which makes conventional gradient-based optimization techniques very inefficient since they rely on these gradients to guide the learning process.

<br>

# The near term

In all honestly, while popular at the moment, I imagine quantum machine learning will go through an "AI winter" similar to what happened from 1974 to 1980. Lots of time, focus and research into these problems will need to happen before the innovation to make these techniques work will come. 

I believe the poor performance in my particular case is because the network is running on a noisy intermediate-scale quantum (NISQ) device, and at the moment these simulators have limitations like limited qubit numbers, short coherence times, and gate errors. But my challenges are also algorithmic, so further tuning of the learning rate, network architecture, data encoding, and other hyperparameters will probably improve training convergence/ All of these factors contribute to the instability of HQNNs.

Because quantum computing is inherentlty susceptible to error via noise because of the delicate nature of quantum states. A phenomenon known as "decoherence" causes information loss. If you're training a HQNN on a noisy quantum device, the errors can affect a multitude of things. From the gradient calculations, which lead to slow or non-convergent training, to the gradient scaling to almost zero because of the vast regions in the quantum optimization landscape. In these regions, standard gradient-based optimization techniques struggle to find a direction for improvement. A training process might get stuck in these plateaus, making it seem as though the network isn't learning.

The way in which classical data is encoded into quantum states can be another source of instability. Some encoding strategies might not preserve the nuances of the data or might be sensitive to small perturbations. And while classical neural networks are also sensitive to hyperparameters like learning rate or architecture specifics, HQNNs introduce additional hyperparameters related to quantum operations, encoding strategies, and error mitigation techniques. The training dynamics can be quite sensitive to these parameters.

Most if not all of these errors are reflected in my networks results, as when running the algorithm multiple times, on one run I could get a 20% classification accuracy, on the second run I could get a 45%, and on the third run I could get a 80% accuracy and so on. It seems to me they are just too unstable to be used for anything other than research at this point. For current industry problems we need scalability and generalization, and it looks like quantum deep learning algorithms are just not ready yet.
