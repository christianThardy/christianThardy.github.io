## Deep Learning Can Solve Non-Trival Problems in NLP

Deep learning is thought of as the ability of a class of algorithms to learn a hierarchical set of representations from lots of data, which means it can learn low, mid and high level features. You specify the input/output and your networks optimization algorithm updates the weights that determine how each input feature will affect the output prediction. 

<img src = "https://user-images.githubusercontent.com/29679899/101258199-ef4f4c80-36ee-11eb-9720-28d24f1b7d16.png" width="455" height="350">

These updates are analogous to a programmer coding a bunch of rules to solve a problem, but with deep learning, sometimes you can end up with code better than what any programmer could write.

We can narrow this down with something more tangible, so let’s use named entity recognition as an example.

We can rely on 3 strengths of applying deep learning to this problem space. 

1. NER benefits from the non-linear transformations made in deep layers, which generates non-linear mappings from input to output. Compared to linear models, deep learning based models are able to learn complex and intricate features from data via nonlinear activation functions.

2. Deep learning saves significant effort on designing NER features. Traditional feature-based approaches require a considerable amount of engineering time. Deep learning models are extremely effective in automatically learning useful representations and the underlying factors from raw text. 

3. Deep neural named entity recognition models can be trained in an end-to-end paradigm by way of gradient descent. This allows us to design NER systems from complex annotation policies.

### Theory

Let’s say the network probably looks at the character level of words...

<img src = "https://user-images.githubusercontent.com/29679899/101260507-8d96de80-36fe-11eb-835f-d2eb04caff8e.jpg" width="645" height="300">

...before looking at whole words...

![SpaCy%20sentence%20for%20blog](https://user-images.githubusercontent.com/29679899/101266353-1a03ca00-371c-11eb-8483-2e775ec92c22.png)

...and arriving at the deepest layer to learn the most distant relationships between the parts-of-speech and the dependency relationships between words and phrases.

Theoretically, many layers should enrich the levels of the features.

In a way, deep learning goes against the intuition that linguistics gives us. That complex linguistics features like parts-of-speech and dependencies should be all we need to parse and extract the differences between concepts in text and language to learn complicated contextual information. Most linguists would be up in arms over this (because it breaks their entire universe), but I am prepared to assert that this is not the case in the field and these features are not enough on their own to help computers with meaning representation. 

The line of thinking that they are is similar to what symbolic expressions were in AI from the mid-1950s to the late-1980s. That cognition is symbolic, and our thoughts are symbolic just because that's what we see, say and imagine.

This turns out to not be the case, as it's more likely that thoughts are a series of abstractions that follow complex sequences in a process that we cannot yet formally explain with strong mathematical intuitions.

Maybe simple linguistic feature sets can be used for very simple use-cases or to optimize some process downstream, but semantics are hard which makes it difficult to reduce the problem to a limited feature set. For example, in more conventional NLP problems, if you're topic modelling you have an overall unambiguous theme, like Apple computers, then you look for all sentiment or mentions or entities relating to this one theme. But if you're working with a wide range of terms that represent themes from a wide range of classes, this makes it hard to pin down uniformity across tasks like keyword extraction using conventional methods. 

I think it would be fair to say that a method that can generalize and be penalized when it gets things wrong would actually be a better way to frame the problem. The ideal approach would be to integrate symbolic and connectionist views of the world, but the one that lacks biological rigor cannot survive without the other.

But it makes sense to me why this view of the world was so popular. There’s a continuing bias towards computers needing to be 100% correct all the time because classical computer science is a discrete math and by that logic everything needs to be 100% correct to function.
Trying to come up with a rule for everything falls in line with how we think about computer science and its OCD nature of trying to account for every single detail of a program within a set of convergence bounds.   

When it comes to applying machine learning to complex problems, there’s usually nothing exact about complex things and we need to consider that we want to automate processes that are often nuanced. So some problems need generalization, which makes a 100% rules based view of the world economically unfeasible and is probably why projects like the Semantic Web cannot secure funding.

But are building neural networks as easy as adding more layers to the network?

More layers do allow the network to learn more fine grained features, but in reality training gets worse, not better when you stack lots of layers, making it harder for the network to choose parameters from the feature space.

### Application

The best approach to this would be to use a network architecture that can take advantage of having many layers and not be affected by the vanishing gradients during gradient descent as the network gets deeper.

One way to frame the NER problem is to train your task using the transition-based, bloom embedding residual CNN state-machine provided by spaCy.

This transition-based state-machine takes a page from the dependency parsing community's use of shift-reduce parsers by using 3 transition operations(actions) to change the state of the input sequence.

`SHIFT, REDUCE, OUTPUT.` 

This “state machine”...

<img width="579" alt="algorithm" src="https://user-images.githubusercontent.com/29679899/101258880-730b3800-36f3-11eb-8b9c-2cfcd41c3911.PNG">

...uses 3 stack data structures (one called the buffer(also sometimes called queue). Items from the buffer go to the stack, and from the stack to the output to move (`SHIFT`) the first word in a sentence from the head of the buffer (contains words to be processed) to the stack (where the words will be further processed). 

So we start with an empty stack, all words from the sentence are on the buffer and we have no entities affixed to the words.

<img width="280" alt="stackbuffer" src="https://user-images.githubusercontent.com/29679899/101259265-ced6c080-36f5-11eb-9db5-81f6137b1bbb.PNG">

During this collection and parsing of the text, the resCNN is making the decision of which named entity label to assign to which words or if a label should be assigned to any particular word at all. 

<img width="321" alt="stackbuffer2" src="https://user-images.githubusercontent.com/29679899/101259759-d77cc600-36f8-11eb-99c0-30bd1b467ef7.PNG">

When assigning and fixing the label to a token, the resCNN incrementally parses the input with bloom embeddings and defines a probability distribution over each action given the current contents of the stack, buffer, output and the history of each action taken to predict the sequence of actions the 3 operations should take to parse the input sentence.

When a token or collection of tokens from a sentence in the buffer are moved to the stack, the resCNN will decide if these tokens belong to a particular entity during the `REDUCE` step, from which the tokens are popped from the top of the stack onto the `OUTPUT` stack and given an entity label.

Then it looks at the next word in the sequence to repeat these actions. The algorithm completes when the buffer and first stack are both empty. 

Words can also be moved from the buffer directly into the output stack when the word should not be labeled. 

So we `SHIFT` once and `[Christian]` moves from the buffer to the stack. We `SHIFT` again and `[Hardy]` joins `[Christian]` on the stack as `[Christian, Hardy]`. The `REDUCE` operation is then used to put `[Christian Hardy]` on the `OUTPUT` stack and adds the label PERSON as `[(Christian Hardy)-PER]`.

<img width="716" alt="parser" src="https://user-images.githubusercontent.com/29679899/101257477-bcf11f80-36ed-11eb-9a0a-05c26a264ce4.PNG">

Spacy bakes in a way for the algorithms to say some actions are valid and some are invalid, which is useful because if we show the algorithm training examples that we do not want tagged, the final model will not make predictions on words it's not sure about. 

So when it's calculating which action to take next, if one entity is predicted upstream and we move to the next potential entity to be tagged and the word is scored as "o" (outside of an entity aka this word is not an entity) the algorithm knows this word is not a valid entity and will not make a label prediction. This means the sequence that's predicted should always be a valid sequence because we have a way to block out noise. 

It's worth noting that this framework is very flexible because the resCNN can define multiple configurations of the transition operations. After the state machine is trained on the labeled examples, it’s REDUCING from the stack to the output, making hundreds of thousands of predictions on new input sentences based on different sequences of word combinations, and updating the learning gradient if an entity label is wrong, optimizing itself so it can make better predictions. Which is why this method is able to come up with good solutions for structured prediction tasks. 

For example, if you have a set of text where the context is very free and relational patterns are hard to extract via their parts-of-speech or their dependencies, framing the problem this way allows you to flexibly calculate features and tune the prediction task to the domain text so that you can read a sentence from it, maintain some sort of state, manipulate that structure with some universal set of actions and penalize the algorithm when it gets things wrong so it can learn to make the right associations.

Earlier I mentioned something called a resCNN (residual convolutional neural network) that predicts the label to fix to our words. They are based on the concept of a Resnet, made popular in 2015.

![ResNet-Variations_2x-700x323](https://user-images.githubusercontent.com/29679899/101259975-853ca480-36fa-11eb-99be-7bafe389f68e.png)

Because of them we can train a deeper network and the learning gradient has a shortcut that it can take in the learning process during backpropagation and that gradient will not vanish.

The resCNN solves this problem by adding skip connections which take the activations from early layers, skips a couple and then feeds them to layers that are deeper in the network. By doing this you can effectively train deeper networks and learn better features.

<img src = "https://user-images.githubusercontent.com/29679899/101259991-ae5d3500-36fa-11eb-91fa-a09c9d04034a.png" width="545" height="360">

Activation functions are necessary to squish the input received from the previous hidden layer down into a small range of values. Without this, we would not be able to learn non-linear relationships. They elongate and squash the layers in space... 

<p align="center">
  <img src="https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/1layer.gif" width="345" height="370">
</p>

...which is the layer trying to approximate the distribution of the input data's topological properties... 

<p align="center">
  <img src="https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/topology_2D-2D_train.gif" width="345" height="370">
</p>

...and without skip connections, it's harder for the deeper layers to learn from linear activation functions during gradient descent.

Non-linear activations help with this somewhat, but essentially they all have some disadvantage of slow convergence, vanishing/exploding gradients, they’re computationally expensive, and some are only suitable for output layers. 

Maybe we need a better optimization algorithm?

<p align="center">
  <img src="https://media.giphy.com/media/LpgBPYLnrMlhUYUrdp/giphy.gif">
</p>

In any case, we can explore this with a sensitivity analysis and look at gradient descent's effect on the input space to maybe interpret what's happening with the data as it goes through deeper layers. In any case, skip connections help give us an advantage. 

During the first step of learning, to give the network a representation of the text so it can learn about its structure, we use dense Bloom embeddings to represent words in context to other words, because the distribution of target words or phrases that we would like to tag are highly dependent on the company they keep. 

This allows us to learn about the input distribution of the incoming sentence from a set of compressed sentences (to maintain network efficiency) and allows the buffer and stack to interact with each other and efficiently check whether a token is a member of its respective set, as the final layer in the resCNN makes its prediction of the state of the input sequence operations. The lower the dimensionality of our inputs, the less memory we use because parsing web-scale text is computationally taxing and the network can make faster decisions. This representation is also easily reversible, so the output vectors can be mapped to their original tokens at prediction time.

### References

Neural architectures for named entity recognition

A survey on deep learning for named entity recognition

Complex linguistic features for text classification: a comprehensive study
