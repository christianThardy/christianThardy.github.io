# Quantum machine learning will change machine learning

The findings of the South Korean physics researchers have been verified, and a significant physics discovery of our era may has not occured yet. But the potential for superconductors that function at ambient pressure and room temperatures would fundamentally change the qubit's role in quantum computing. This change could aid in scalability, although higher temperatures can still affect quantum inaccuracies. Nevertheless, room temperature superconductors represent a crucial element of the quantum conundrum, simplifying the creation of hybrid systems capable of alternating between classical and quantum computation tasks.

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

Imagine the bits (a binary digit and is the smallest unit of data that a computer can process and store) of a classical computer as a simple light switch in your home; they are either in the OFF (0) or ON (1) position. In the world of regular computing, this binary system forms the foundation of all operations.

Now, picture a qubit as a dimmer switch instead of a simple light switch. The dimmer switch not only has an ON and OFF position but also various stages in between, allowing for a wide range of light levels. This is a lot like how qubits work. They can represent not just 0 or 1 but also a superposition of both 0 and 1 simultaneously, or a range of values between 0 and 1. This ability to be in multiple states at once allows quantum computers to process a vast amount of information simultaneously, leading to potentially enormous computational power.

To understand quantum gates, imagine a toy train set. Classical gates (boolean logical operators in classical programming like AND, OR, NOT) in regular computing are akin to the set's basic pieces, which allow the train (data) to move in specific paths, join tracks, or reverse direction. They perform specific logical operations on the bits.
<br/>

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/47d9e65f-5c28-4d48-9d70-d2a5edc73937">
</p>

<br/>

Quantum gates are like magical pieces in this train set that can rotate the tracks, twist them into three-dimensional shapes, or even create tunnels that the train can simultaneously travel through and not travel through. These "magic" pieces perform specific operations on qubits, manipulating them in complex ways, which include superposition and entanglement (where qubits become interconnected and the state of one affects the others).

Now, take the concept of deep learning, a form of artificial intelligence where classical computers imitate the way our brains think, forming networks to "learn" from data. Picture it like a very intricate train set with many tracks, switches, and stations representing the complex relationships in the data.

In the context of quantum deep learning, replace this classical train set with our magical quantum train set, equipped with dimmer switches (qubits) and magical pieces (quantum gates). This allows for an extraordinarily sophisticated network that can explore multiple pathways simultaneously, make connections more quickly, and potentially unlock solutions that classical systems might never find.
<br/>

# Limitations

With the limitations of near-term quantum hardware potentially being a thing of the past soon as physics research advances, the maturity of classical algorithms is another matter that quantum algorithms will need to address. 

Let's walk through an example. I have the make moons dataset and BLANK BLANK BLANK. We can illustrate how effective classical machine learning algorithms are, how much better neural networks are, and how far quantum neural networks need to go before we can begin using the technology in a meaningful way.
<br/>

# Limitation understanding

The make moons dataset is a great tool for when you want to experiment with or visualize algorithms that deal with complex, non-linear relationships. Imagine looking up at the night sky and seeing two crescent moons touching at their tips. 

<br/>

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
Accuracy of logistic regression model: 88.47%
```

<br/>

# Multi-layer perceptron neural network

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/f8d79691-e28c-49c4-a371-54e16c09ce66" width="440" height="330">
</p>

```python 
Accuracy of neural network model: 94.99%
```

<br/>

# Quantum neural network

<p align="center">
<img src="https://github.com/christianThardy/christianThardy.github.io/assets/29679899/3a99e2aa-43d3-42aa-8e54-4f9cd4782e03" width="400" height="300">
</p>

```python
Accuracy of neural quantum model: 45.69%
```
