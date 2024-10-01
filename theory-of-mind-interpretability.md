# Theory of mind and GPT models

Mechanistic interpretability allows us to reverse engineer the inner workings and representations learned by neural networks into understandable algorithms and concepts that provide a granular, causal understanding of neural networks.

Given my current focus on LLMs and my interest in psychology, I've been asking myself how do decoder-only language models perform theory of mind tasks. I have a theory that some simplification of abstract reasoning tasks like the theory of mind (ToM) task can be interpreted from the inner mechanisms of a GPT model to understand its internal representations of ToM tasks. If the circuit (algorithm) that completes this task can be reverse engineered, what makes that possible in a GPT-2 model?

Humans are capable of making inferences about the mental state of characters in a ToM sentence. These inferences require syntactic or prepositional logic, but what else? Let's first explore the linguistic phenomena of **First-Order Logic** (FOL), **Semantics** and **Pragmatics**.

<br>

# First-Order Logic

Sentences where you can make inferences require FOL, semantics and pragmatics. It provides a framework for representing and manipulating the meaning of sentences in a structured and formal way, also helps in mapping syntactic structures of natural language sentences to their corresponding semantic representations.

Let's take the sentence: *In the room there are John, Mark, a cat, a box, and a basket. John takes the cat and puts it on the basket. He leaves the room and goes to school. While John is away, Mark takes the cat off the basket and puts it on the box. Mark leaves the room and goes to work. John comes back from school and enters the room. John looks around the room. He doesn’t know what happened in the room when he was away. John thinks the cat is on the...'*

In the context of ToM, to make the correct prediction *basket*, the model needs to understand:

  - **Entities:** Mark, John, cat, basket and box.

  - **Properties and Relations:** John puts the cat on the basket, John remembering where he put the cat, John's expectation that the cat will be on the basket.

  - **Mental States:** John's belief and expectation that the cat will be on the basket after he returns.

<br>
  
FOL helps in maintaining the context and managing the state of a conversation by representing a dialogue state in logical terms. For example:

  - Take(John, cat)\
    PutOn(John, cat, basket)\
    Leave(John, room)\
    GoTo(John, school)\
    Take(Mark, cat)\
    PutOn(Mark, cat, box)\
    Leave(Mark, room)\
    GoTo(Mark, work)\
    ComeBack(John)\
    Enter(John, room)\
    Thinks(John, On(cat, basket))\
    NotKnow(John, HappenedDuring(AwayTime))

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/ff692b8a-8f6e-4e8a-9abf-68b36fe27d2a" width="800"/>
</p>

<br>

This structure allows us capture meaning because of relationships and quantifiers, which are essential for capturing the nuanced meanings and mental states involved in ToM. It captures the internal structure of propositions, such as the belief that Jane expected John to bring something specific.

So its possible that at some level, ToM prediction in LLMs aligns with first-order logic due to a models ability to represent complex relationships between entities and their properties, which are crucial for understanding and predicting human mental states and behaviors.

<br>

# Semantics

Semantics refers to the study and representation of meaning in language. Semantics deals with how words, phrases, and sentences convey meaning, and how this meaning is interpreted by humans and machines. It focuses on the inherent meaning of words and sentences. Semantics encompasses a lot ranging from compositional semantics, semantic similarity and even word embeddings/distributional semantics. 

For example, to understand the semantics of the sentence we need to know it implies something was supposed to be on the plate (likely food), recognize that Jane's sigh is expressing dissappointment or frustration, recognize that when Jane is thinking she is considering her internal state and expectations because it was her birthday, and use context clues to infer that what John was supposed to bring on her birthday is related to the empty plate and Jane's sigh. 

To do all of this we need to identify all entities and actions in the sentence.

  - Entities: Jane, John, the plate.
  - Actions: looked, sighed, thought, bring.
    
  - Extract Relationships and Properties:
      - The plate is empty: E(plate)  
      - Jane's actions and mental state: Sigh(Jane), Think(Jane, ifonlyJohnhadrememberedtobringthecake)

<br>

In the context of semantics, ToM prediction requires extracting the meaning of a sentence, including understanding entities, their properties, and relationships, which is the core goal of semantic parsing. Semantic parsing can help with understanding context and inferring implied meanings, which is essential for accurate ToM predictions. ToM prediction also involves understanding and representing complex mental states and expectations, which require a structured form that semantic parsing provides. LLMs can understand the underlying meaning and context, allowing them to predict that the missing word is *cake*. This involves both understanding the literal content and inferring the mental states and expectations of the characters.

<br>

# Pragmatics

Pragmatics, usually a key concept in semantics, is focused on how context influences the interpretation of meaning in language. This includes factors like speaker intent, conversational implicature, and situational context. To predict the final word in the example sentence sequence, a model must understand not just the literal meaning of the words but also Jane's mental state, her expectations, and the context in which she is making the statement.

To obtain contextual understanding we need to know situational context, so an empty plate, a sigh, it being Jane's birthday helps infer that something was expected on the plate on this day. A speakers intention and beliefs, so understanding Jane’s disappointment and what she believes John was supposed to bring, and we need the ability to infer the most likely item that fits Jane’s expectation and the context (e.g., a specific food item like "cake").

ToM prediction heavily relies on the context to make sense of the mental states and intentions behind the words, and the final word prediction is based on implied meanings and inferred intentions, which are central to pragmatics. Pragmatics encompasses understanding social interactions, cognitive states, understanding that others have mental states, beliefs, desires, intentions, and perspectives—that are different from one's own, which are key to ToM.

The remainder of this work will specifically focus on how GPT models will implement this task and in the end understand in a tractable way, the mechanisms responsible for completing the task across many different heuristics and metrics.

<br>

# Theory of Mind Circuit Discovery

The model used in this analysis is from Stanford's Center for Research on Foundation Models (Stanford CRFM) family. The *eowyn-gpt2-medium-x777* model to be exact. It is a decoder-only transformer that has 23 layers and 15 attention heads per attention layer. The focus is the circuit that successfully models the ToM task by understanding the behavior of the attention heads and mlps.

In terms of the internal mechanisms of a language model, a feature is a property of the input that humans can understand and is represented in a model's activation (the tokens from the ToM sentence). A circuit informs us of how these features are extracted from the input and then processed by the model to implement specific language model behaviors (e.g., reasoning), which gives us an algorithmic understanding of the models reasoning.

Humans make predictions about others' thoughts and feelings —a key component of ToM— through a combination of neurological processes and behavioral observations. These processes are intricate and involve multiple steps, both at the neural and cognitive levels. At the level of a decoder-only transformer model, we can first broadly understand ToM prediction for this specific sentence structure through an interpretable algorithm largely dependent on John's mental state of where he put the cat: 

       - Consider events the subjects have witnessed.
       - Consider the location of objects based on the subject's last knowledge.
       - Ignore events that occurred while the subject was absent.
       - Predict subjects belief about the object's location based on the last event they witnessed.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2755018d-dd41-4bf7-adb7-d1f3ed087310" width="800"/>
</p>

<br>

### ToM Circuit Discovery: Identify Relavant Activations & Layers

Thanks to <a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" title="lesswrong.com" rel="nofollow">nostalgebraist</a> we have the logit-lens. So we can determine how language models refine their predictions across layers. The approach will be applied to interpret activations in features, but first to circuit discovery.

Causal interventions in the context of this analysis give way to techniques so that model components can be manipulated to understand or influence how different parts of the model contribute to the final output. In order to evaluate how model performance changes when performing causal interventions, we need a metric to measure model performance. 

The metric used here will be the logit difference, the difference in logit between the name of the actual location of the object and the name of the believed location of the object: `logit(basket) - logit(box)`.

When deconstructing the residual stream, the logit-lens looks at the residual stream after each layer and calculates the logit difference from there. This simulates what happens if we delete all subsequence layers. The final layernorm are applied to the values in the residual stream and then project them in the logit difference directions.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/40724e17-b54b-4d1b-aeff-0cdec72935a4" width="1700"/>
</p>

<br>

What's interesting is that the model shows almost no capacity to handle the task until we get to layer 22. And then—boom—attention layer 22 kicks in and almost all the performance happens there, and then things get worse after that layer. It’s not just a smooth upward trajectory; there’s a clear peak followed by a clear descent.

So, what’s going on here? It’s a strong signal that layers 22, 23, and 24 are doing something really specific—writing to the residual stream in a way that allows the model to solve the ToM task. This insight can help us narrow the investigation and gives a clear direction: we need to figure out what kind of computation these layers are performing. It opens up exciting questions: How do attention layers (moves information around) compare with MLPs (processes information) in their contribution? And within those attention layers, which heads are doing the heavy lifting?

In terms of narrowing, is where things get really fun, and now you can start isolating the mechanisms and digging into specific computations, which will give real insights into how the model solves ToM.

Repeating the previous analysis, but for each layer by activation reveals how to begin the narrowing process.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/593f2793-f33a-4932-94be-d59a5d03a4d4" width="1700"/>
</p>

<br>

It looks like only the attention layers matter here. The ToM task, similar to the IOI task is all about moving information around, pulling John's believed location of the cat into focus while ignoring the actual location of the cat. While there is minimal complex processing by the MLPs which warrents investivation, the emphasis is on the attention.

What’s particularly interesting is that attention layer 22 gives us a big boost in performance, but then things take a turn— MLP layer 22 and attention layer 23 and subsequent MLP layers actually make things worse. So, the attention mechanism is crucial, but there's a point where additional layers start to hurt more than help. This kind of dynamic tells us something important about how information flows through the model and where it can break down.




<br>
<br>

When evaluating the importance of each model component for the ToM task, we can see interesting behavior in the attention patterns of the 14th attention head in layer 22.



<br>


