# Theory of Mind and GPT models

<a href="https://www.neelnanda.io/mechanistic-interpretability/quickstart" title="www.neelnanda.io" rel="nofollow">Mechanistic interpretability</a> allows us to reverse engineer the inner workings and representations learned by neural networks into understandable algorithms and concepts that provide a granular, causal understanding of neural networks. We can conceptualize this as some path inside a model that goes from the input to the output where we can trace which paths in the model matter, and decompose a path between different parts of the model that we expect to be interpretable. 

Given my current focus on transformer-based LLMs, theory of mind (ToM), and mechanistic interpretability, I've been asking myself many core questions:

How exactly do decoder-only language models (DOLMs) perform and *solve* ToM tasks? What is the model doing, and what algorithms is the model using when it is performing and solving ToM?

Is it appropriate to evaluate DOLMs at the level of a psychologist analyzing a human subject to determine its level of attained ToM? One way they think about this is through frameworks like <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6974541/" title="ncbi.nlm.nih.gov" rel="nofollow">ATOMS</a> (Abilities in Theory of Mind Space), which categorizes concepts like beliefs, intentions, desires, emotions, knowledge, and percepts. But is this the best approach for understanding model behavior, or can we gain more clarity by zooming in and analyzing the internal mechanisms that enable ToM capabilities? 

What’s the real burden of proof for ToM in DOLMs? If we find a clear algorithmic process that they implement to solve ToM tasks in a way that heavily relies on the structure of language, does that automatically mean the model isn’t really engaging in ToM, or could it be that this is the way models represent the abstract reasoning that ToM requires? Another key question is whether ToM tasks are solvable purely by leveraging syntactic structures and linguistic properties via compositionality. If compositionality is exploited by the model to solve ToM, are these just "shortcuts" that "give answers away", or are they core features that DOLMs rely on to perform *and* solve these tasks? 

Do we have an interpretable algorithm that clearly explains how humans solve ToM tasks that is outside of the scope of combining prior knowledge with observed behaviors and contextual nuances (intentionally ignoring emotions and cultural norms) in the human brain? The left language-dominant prefrontal cortex encodes semantic information during speech processing. These neural responses are dynamic, reflecting the contextual meanings of words rather than fixed memory representations, which reveals a detailed organization of semantic representations at the cellular level during language comprehension. Which shows us that our brains use compositionality[<a href="https://www.nature.com/articles/s41586-024-07643-2" title="Jamali" rel="nofollow">3</a>] to process language.

There's always the argument of how brittle data can be. Even datasets of hundreds of billions of tokens will not cover every word, clause or preposition that may be encountered in the future. New, unseen ToM data could always "break" the model—so it would struggle on data it hasn’t been explicitly trained on. 

But even beyond that, do the mechanisms for performing and solving ToM in DOLMs remain consistent across different samples? While it’s likely that updating the training data could lead to short-term improvements, the broader challenge of evaluating ToM may be harder to fully resolve due to our understanding of ToM and the inherent limitations of DOLMs and datasets.

While I'm skeptical about why models are performing ToM or are not performing ToM, my aim to show that some of the abstract reasoning involved in ToM tasks can be simplified into an interpretable algorithm (or circuit) and mechanisms derived from the internal representations of a DOLM. By breaking down these representations and contextualizing them, we might better understand how these models structure and engage with ToM tasks.

<br>

# First-Order Logic

Humans are capable of making inferences about the mental state of characters in a ToM sentence. At a conceptual level, these inferences require syntactic or prepositional logic, but what else? Let's explore the linguistic principles of **First-Order Logic** (FOL), **Semantics** and **Pragmatics**.

Sentences where you can make inferences require FOL, semantics and pragmatics. It provides a framework for representing and manipulating the meaning of sentences in a structured and formal way, and also helps in mapping syntactic structures of natural language sentences to their corresponding semantic representations.

Let's take this false belief passage: *'In the room there are John, Mark, a cat, a box, and a basket. John takes the cat and puts it on the basket. He leaves the room and goes to school. While John is away, Mark takes the cat off the basket and puts it on the box. Mark leaves the room and goes to work. John comes back from school and enters the room. John looks around the room. He doesn’t know what happened in the room when he was away. John thinks the cat is on the...'*

In the context of ToM, to make the correct prediction `basket`, the model needs to understand:

  - **Entities:** Mark, John, cat, basket and box.

  - **Properties and Relations:** John puts the cat on the basket, John remembering where he put the cat, John's expectation that the cat will be on the basket.

  - **Mental States:** John's belief and expectation that the cat will be on the basket after he returns.

<br>
  
FOL helps in maintaining the context and managing the state of a conversation by representing a dialogue state in logical terms. For example:

  - Take(*John, cat*)\
    PutOn(*John, cat, basket*)\
    Leave(*John, room*)\
    GoTo(*John, school*)\
    Take(*Mark, cat*)\
    PutOn(*Mark, cat, box*)\
    Leave(*Mark, room*)\
    GoTo(*Mark, work*)\
    ComeBack(*John*)\
    Enter(*John, room*)\
    Thinks(*John, On(cat, basket*))\
    NotKnow(*John, HappenedDuring(AwayTime)*)

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/ff692b8a-8f6e-4e8a-9abf-68b36fe27d2a" width="800"/>
</p>

<br>

This structure allows us to capture meaning because of relationships and quantifiers, which are essential for capturing the nuanced meanings and mental states involved in ToM. It captures the internal structure of propositions, such as the belief that `John` expects the `cat` to be somewhere specific. 

So its possible that at some level, ToM prediction in DOLMs aligns with first-order logic due to a models ability to represent complex relationships between entities and their properties, which are crucial for understanding and predicting human mental states and behaviors.

<br>

# Semantics

Semantics refers to the study and representation of meaning in language. Semantics deals with how words, phrases, and sentences convey meaning, and how this meaning is interpreted by humans. It focuses on the inherent meaning of words and sentences. Semantics encompasses a lot ranging from compositional semantics, semantic similarity and even word embeddings, distributional semantics and distributed semantics. 

For example, to understand the semantics of the following passage linguistically, we need to identify the entities, actions, relationships, and implied meanings. To do all of this we need to identify all entities and actions in the sentence.

  - **Entities:** John, Mark, cat, basket, box, room
  - **Actions:** takes, puts, leaves, goes, comes back, enters, looks, doesn't know, thinks
    
  - **Extract Relationships and Properties:**
      - **Initial State of Entities:**
        - InRoom(*John*)
          InRoom(*Mark*)
          InRoom(*cat*)
          InRoom(*box*)
          InRoom(*basket*)

      - **John's Actions (Initial):**
          - Take(*John, Cat*)
            PutOn(*John, Cat, Basket*)
            Leave(*John, Room*)
            GoTo(*John, School*)

      - **Mark's Actions (While John is away):**
          - TakeOff(*Mark, Cat, Basket*)
            PutOn(*Mark, Cat, Box*)
            Leave(*Mark, Room*)
            GoTo(*Mark, Work*)

      - **John's Actions (His return):**
           - ComeBack(*John*)
             Enter(*John, Room*)
             LookAround(*John*)
             NotKnow(*John, Actions(Mark)*)
             Think(*John, On(Cat, Basket)*)
               
<br>

In the context of semantics, understanding and interpreting this passage requires extracting the meaning of each sentence, identifying entities, their properties, and relationships—core goals of semantic parsing. Semantic parsing aids in comprehending context and inferring implied meanings, which is essential for accurate ToM predictions. ToM involves understanding and representing complex mental states and expectations. In this example, DOLMs can grasp the underlying meaning and context, allowing them to predict that `John` thinks the `cat` is on the `basket`, even though it is actually on the `box`. This involves understanding both the literal content and inferring the mental states and beliefs of the characters.

<br>

# Pragmatics

Pragmatics, a key concept in semantics, focuses on how context influences the interpretation of meaning in language. This includes factors like speaker intent, conversational implicature, and situational context. To predict the final word in the example passage sequence, a model must understand not just the literal meaning of the words but also John's mental state, his expectations, and the context in which he is making the statement.

To obtain contextual understanding, we need to know the situational context—

`John` placed the `cat` on the `basket` before leaving for `school`, and he is unaware that `Mark` moved the `cat` to the `box` while he was away. 

Understanding John's beliefs and what he expects to find upon his return is crucial. We need the ability to infer the most likely location that fits John's expectation and the context (e.g., the `basket`). This involves recognizing that `John` thinks the `cat` is still where he left it, demonstrating the importance of pragmatics in interpreting language and predicting intended meaning.

<br>

## So What?

These concepts and processes can *help* explain how humans can perform ToM linguistically, but are these concepts or processes mimicked in transformers? 

ToM prediction heavily relies on context to make sense of the mental states and intentions behind the words, and the final word prediction is based on implied meanings and inferred intentions, which are central to pragmatics. Pragmatics encompasses understanding social interactions, cognitive states, understanding that others have mental states, beliefs, desires, intentions, and perspectives—that are different from one's own, which are key to ToM.

The remainder of this work will specifically focus on how a DOLM will implement this task and in the end understand in a tractable way, the mechanisms responsible for completing the task across different heuristics and metrics, and whether or not these high level linguistic concepts are appropriate or not to think about how language models perform ToM.

<br>

# Theory of Mind Circuit Discovery

The model used in this analysis is Gemma-2-2B from Google's family of Gemma models. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/1efe16c2-cf0b-40a3-90df-b190f68b2960" width="250"/>
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/ecec4cba-66c3-4b05-acfc-132c66804021" width="400"/>
<br>
<small style="font-size: 10px;"><a href="https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/" title="Google Developers Blog" rel="nofollow">Google's Gemma, 2024</a></small>
</p>

<br>

It is a decoder-only transformer that has 25 layers and 7 attention heads per attention layer. The broader focus of this analysis is identifying the circuit that successfully models the ToM task, and the narrow focus is indentifying that circuit by understanding the behavior of the attention heads, MLPs and residual streams.

In terms of the internal mechanisms of a language model, a **feature** is a property of the input that humans can understand and is represented in a model's activation (the tokens from the ToM passage). A **circuit** informs us of how these features are extracted from the input and then processed by the model to implement specific language model behaviors (e.g., reasoning), which gives us an algorithmic understanding of the models reasoning. First we understand the features, use those features to understand the circuits which connect those features and once we understand more circuits we can understand the model.

Humans make predictions about others' thoughts and feelings —a key component of ToM— through a combination of neurological processes and behavioral observations. These processes are intricate and involve multiple steps, both at the neural and cognitive levels. At the level of a decoder-only transformer model, we can first broadly begin to understand ToM prediction for this specific passage through a simple interpretable algorithm largely dependent on John's mental state of where he put the cat: 

       - Consider events the subjects have witnessed.
       - Consider the location of objects based on the subject's last knowledge.
       - Ignore events that occurred while the subject was absent.
       - Predict subjects belief about the object's location based on the last event they witnessed.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2755018d-dd41-4bf7-adb7-d1f3ed087310" width="800"/>
</p>

<br>

### ToM Circuit Discovery: Identify Relavant Layers & Activations to the Task

Thanks to <a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" title="lesswrong.com" rel="nofollow">nostalgebraist</a> we have the logit-lens. So we can determine how language models refine their predictions across layers. The approach will be applied first to interpret layers and activations, and then to features and circuit discovery.

Causal interventions in the context of this analysis give way to techniques so that model components can be manipulated to understand or influence how different parts of the model contribute to the final output. In order to evaluate how model performance changes when performing causal interventions, we need a metric to measure model performance. 

The metric used here will be the logit difference, the difference in logit between the entity of the believed location of the object and the entity of the actual location of the object to gauge the accuracy of the models answers: `logit(basket) - logit(box)`.

We can use the same circuit finding framework as the <a href="https://arxiv.org/pdf/2211.00593" title="Interpretability In The Wild: A Circuit For Indirect Object Identification In GPT-2 Small" rel="nofollow">Indirect Object Identification</a> (IOI) task as a basis for understanding ToM, as indirect object-subject entities can be mapped to original-new location entities.

<br>

```python
# Decoder-only model performing IOI
Tokenized prompt: ['<bos>', 'After', ' John', ' and', ' Mary', ' went', ' to', ' the', ' store', ',', ' John', ' gave', ' a', ' bottle', ' of', ' milk', ' to']
Tokenized answer: [' Mary']
Performance on answer token:
Rank: 0        Logit: 18.09 Prob: 70.07% Token: | Mary|
Top 0th token. Logit: 18.09 Prob: 70.07% Token: | Mary|
Top 1th token. Logit: 15.38 Prob:  4.67% Token: | the|
Top 2th token. Logit: 15.35 Prob:  4.54% Token: | John|
Top 3th token. Logit: 15.25 Prob:  4.11% Token: | them|
Top 4th token. Logit: 14.84 Prob:  2.73% Token: | his|
Top 5th token. Logit: 14.06 Prob:  1.24% Token: | her|
Top 6th token. Logit: 13.54 Prob:  0.74% Token: | a|
Top 7th token. Logit: 13.52 Prob:  0.73% Token: | their|
Top 8th token. Logit: 13.13 Prob:  0.49% Token: | Jesus|
Top 9th token. Logit: 12.97 Prob:  0.42% Token: | him|
Ranks of the answer tokens: [(' Mary', 0)]
```

```python
# Decoder-only model performing ToM
Tokenized prompt: ['<bos>', 'In', ' the', ' room', ' there', ' are', ' John', ',', ' Mark', ',', ' a', ' cat', ',', ' a', ' box', ',', ' and', ' a', ' basket', '.', ' John', ' takes', ' the', ' cat', ' and', ' puts', ' it', ' on', ' the', ' basket', '.', ' He', ' leaves', ' the', ' room', ' and', ' goes', ' to', ' school', '.', ' While', ' John', ' is', ' away', ',', ' Mark', ' takes', ' the', ' cat', ' off', ' the', ' basket', ' and', ' puts', ' it', ' on', ' the', ' box', '.', ' Mark', ' leaves', ' the', ' room', ' and', ' goes', ' to', ' work', '.', ' John', ' comes', ' back', ' from', ' school', ' and', ' enters', ' the', ' room', '.', ' John', ' looks', ' around', ' the', ' room', '.', ' He', ' doesn', '’', 't', ' know', ' what', ' happened', ' in', ' the', ' room', ' when', ' he', ' was', ' away', '.', ' John', ' thinks', ' the', ' cat', ' is', ' on', ' the']
Tokenized answer: [' basket']
Performance on answer token:
Rank: 0        Logit: 28.59 Prob: 63.25% Token: | basket|
Top 0th token. Logit: 28.59 Prob: 63.25% Token: | basket|
Top 1th token. Logit: 27.91 Prob: 32.20% Token: | box|
Top 2th token. Logit: 24.56 Prob:  1.13% Token: | table|
Top 3th token. Logit: 23.90 Prob:  0.58% Token: | floor|
Top 4th token. Logit: 23.69 Prob:  0.47% Token: | cat|
Top 5th token. Logit: 23.69 Prob:  0.47% Token: | bed|
Top 6th token. Logit: 23.24 Prob:  0.30% Token: | desk|
Top 7th token. Logit: 22.92 Prob:  0.22% Token: | ground|
Top 8th token. Logit: 22.12 Prob:  0.10% Token: | top|
Top 9th token. Logit: 22.10 Prob:  0.10% Token: | shelf|
Ranks of the answer tokens: [(' basket', 0)]
```

<br>

In the IOI task the model distinguishes between indirect and direct objects to predict the name that isn’t the subject of the last clause. In the ToM task, it distinguishes between believed locations of objects and actual locations of objects to understand scenarios involving actions and their sequences to predict the original and new locations of an object.

When deconstructing the residual stream, the logit-lens looks at the residual stream after each layer and calculates the logit difference from there. This simulates what happens if we delete all subsequent layers. The final layernorm are applied to the values in the residual stream and then projected in the logit difference directions.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/40724e17-b54b-4d1b-aeff-0cdec72935a4" width="1700"/>
</p>

<br>

What's interesting is that the model shows almost no capacity to handle the task until we get to layer 22. And then—boom—attention layer 22 kicks in and almost all the performance happens there, and then things get worse right after layer 23. It’s not just a smooth upward trajectory; there’s a clear peak followed by a clear descent after layer 24.

So, what’s going on here? It’s a strong signal that layers 22, 23, and 24 are doing something really specific—writing to the residual stream in a way that allows the model to solve the ToM task. This insight can help us narrow the investigation and gives a clear direction: we need to figure out what kind of computation these layers are performing. It opens up exciting questions: How do attention layers (move information around) compare with MLPs (processes information) in their contribution? And within those attention layers, which heads are doing the heavy lifting? What's going on in the residual stream exactly?

This is where things get really fun. When narrowing down the problem, we can now start isolating the mechanisms and digging into specific computations, which will give real insights into how the model performs ToM.

Repeating the previous analysis, but for each layer by activation reveals how to begin the narrowing process.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/593f2793-f33a-4932-94be-d59a5d03a4d4" width="1700"/>
</p>

<br>

It looks like only the attention layers matter here. The ToM task, similar to the IOI task, is all about moving information around, pulling John's believed location of the cat into focus while ignoring the actual location of the cat. While there is minimal processing by the MLPs that matter (perhaps some level of understanding context is processed here), which warrents investivation, the emphasis is on the attention.

What’s particularly interesting is that attention layer 22 gives us a big boost in performance, but then things take a turn— MLP layer 22 and attention layer 23 and subsequent MLP layers actually make things worse. So, the attention mechanism is crucial, but there's a point where additional layers start to hurt more than help. This kind of dynamic tells us something important about how information flows through the model and where it can break down.

We can break down the output of each attention layer even further by looking at the sum of the outputs of each individual attention head. Every attention layer consists of 7 heads, and each head acts independently and additively to influence the final result.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/3a62bf79-d9cc-4ccc-a455-fcf83bd86e04" width="650"/>
</p>

<br>

Interestingly, while there is positive activity that contributes to the prediction of the ToM task, only a few heads actually matter. Head 3 at layer 0, head 4 at layer 22 and head 3 at layer 23 contribute positively on some range of significance, which explains why attention layer 22 is so crucial for performance. On the flip side, head 7 at layer 18 and heads 5 and 4 at layers 23 and 25 respectively are negatively impacting the model greatly.

These heads correspond to some of the name mover heads (renamed location mover heads for this analysis) and negative name mover heads (renamed negative location mover heads for this analysis) discussed in the paper. There are also other heads that matter positively or negatively but to a lesser degree—these include additional location movers and backup location movers. More on this later.

There are a couple of big meta-level takeaways here. First, even though our model has 7 attention heads in total, we can localize the behavior of the model to just a handful of key heads. This strongly supports the argument that attention heads are the right level of abstraction for understanding the model's behavior.

Second, the presence of negative heads is really surprising—like head 7 at layer 23, which makes the incorrect logit seven times more likely. I don’t fully understand what’s happening there, but the IOI paper touches on a few potential explanations. It's definitely something worth digging into more.

<br>

### ToM Circuit Discovery: The Residual Stream and Attention Analysis

Attention heads are valuable to study because we can directly analyze their attention patterns—basically, we can see which positions they pull information from and where they move it to. This is especially helpful in our case since we're focused on the logits, meaning we can just look at the attention patterns from the final token to understand their direct impact. Specifically, we’ll be looking at the top 3 positive (visualizations for the negative heads were also produced in the analysis) logit attribution heads based on their direct contribution to the logits.

One common mistake when interpreting attention patterns is to assume that the heads are paying attention to the token itself—maybe trying to account for its meaning or context. But really, all we know for sure is that attention heads move information from the residual stream at the position of that token. Especially in later layers, the residual stream might hold information that has nothing to do with the literal token at that position! For example, the period at the end of a sentence might store summary information for the entire sentence. So when a head attends to it, it’s likely moving that summary information, not caring if it ends with punctuation. This makes it hard to asses what the attention heads are doing when tokens are being attended to. 

But at the same time, I think when an attention head is attending to a token, it is accessing abstract information stored at that position.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/c64273c7-5d0b-4efc-bbd1-b0ed05842aa5" width="280"/>
</p>

<br>

In transformer architectures, each token position has a residual stream—a vector that carries forward information as the model processes each layer. We can think of the residual stream as the place where everything communicated from earlier layers are communicated to later layers and must go through this stream. It captures everything the model has "thought" so far, so it will contain everything important going on in the model.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/d1590634-0cb0-42f5-b177-a17ee0203af1" width="280"/>
</p>

<br>

This stream accumulates more than just the token embedding; it also aggregates output from previous attention heads and feedforward networks. Attention heads and mlps read in information from the residual stream, apply edits to the input based on how it functions, and then puts that edited (new) information back into the residual stream. They only read and write from the stream with linear operations (addition), this means the input to any layer can be decomposed to the sum of the output of a bunch of operations that correspond to different mechanisms of every layer inside the transformer.

By the time we get to the later layers, the residual stream should be holding rich, high-level abstractions, like syntactic structures, semantic relationships, or even summaries of entire phrases or sentences of larger text segments. In other words, there should be observable compositionality in the residual stream. Attention heads don't just read from tokens—they read from the residual streams at specific positions and write new information into the residual stream at the target position. This allows them to move contextually rich, abstract information from one position to another, independent of the specific token at those positions.

Going back to our period example, for that period at the end of a sentence, the residual stream at that position might hold a summary of the entire sentence in its residual stream, not just the token embedding of the period itself. It’s a complex, multi-layered representation that has been built up over the entire forward pass over multiple attention blocks and mlps, containing information about syntactic roles, semantic meanings, and even the overall structure of the sentence. So attention patterns are essentially mechanisms for moving these complex representations between positions, often transferring higher-level abstractions like hierarchical structures and temporal sequences. This is why models can handle nested structures and dependencies, which are assumed to be critical for tasks like ToM.

During the process of a layer (mlp or attention) reading from the residual stream, the model has the ability to access all of the information from the previous layers, but the model can choose to focus on a few meaningful directions by aligning the thing they read in with the information they care about, so the mechanism can make sure it mostly gets the information it cares about. So after a set of directions are chosen from the residual stream, they can be written to another mechanism.

The directions in the residual stream space that influence which mechanism moves information where and between mechanisms, are based on the similarity of those residual stream directions and the directions of information in other mechanisms. More on how transformers process information using linear algebra <a href="https://youtu.be/wjZofJX0v4M?si=yzNyY0gmwQ892Z6P&t=747" title="3Blue1Brown" rel="nofollow">here.</a>

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/61e58090-4367-4a9d-b8bb-411fcb5f0e1b" width="280"/>
</p>

<br>

Rather than the input needing to go through every single layer of the network, the model can choose which layers it wants information to go through via the residual stream and what paths it wants to send information to. This is why we can expect model behavior to be kind of localized, so as the input goes through each mechanism, not every piece of the input will receive an activation.

The model is using the residual stream to achieve compositionality between different pieces of information, and its how mechanisms in the model communicate with each other. For example, there could be some attention head in layer 2 that composes with some head in layer 22. Technically this looks like some head in the 1st layer will output some vector to the residual stream, the head in the 2nd layer will take as an input the entire residual stream and mostly focus on the output of the 1st layer and run some computation on it. For any pair of composing pieces in the model, they are completely free to choose their own interpretation of the input, so there's no reason that the encoding of the information between head 0 in layer 0 and head 5 in layer 3 will be the same as the encoding between head 2 in layer 0 and head 3 in layer 1. This means we can expect the residual stream to be very difficult to interpret.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/950b47aa-675d-4a47-82c7-d7ce8b457552" width="7500"/>  
</p>

<br>

So, what’s happening here is the model builds up hierarchical representations of language—phrases within sentences, sentences within paragraphs—and tracks sequences of events, which is particularly important for tasks like Theory of Mind (ToM), where understanding the events, the order of events, character actions and possibly even directional or spatial information is key.  In this framework, attention heads work like routers, directing specific pieces of information to the right places to solve the task. They aren’t just focusing on literal tokens but transferring abstract concepts like *"the last place John saw the cat"*, which aren't tied to any single token but are encoded in the residual stream.

This kind of hierarchical, nested structure in the residual stream is key to solving the IOI task. The task requires the model to parse grammatical roles, like identifying subjects, objects, and indirect objects, and understand their relationships. Similarly, the ToM task requires the model to track what each character knows or believes over time, which means keeping updated representations of these abstract knowledge states in the residual streams.

In any case, it’s easy to get tricked if you think an attention head is just focusing on a literal token. What we should be looking at is the information stored in the residual streams at that position—often abstract concepts or higher-level representations—rather than just assuming the attention head is simply "attending" to the token itself.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/527153fb-f75d-4ec2-a3e3-52a459740d41" width="1000"/>
</p>

<br>

While keeping all of that in mind, when looking at this plot, it’s a good time to start thinking about the algorithm the model might be running. Specifically, for the attention heads with high positive attribution scores, we can see `the` is attending to `basket` with high confidence, particularly the second time basket is referenced, and `box` with lower confidence. How might this head’s behavior be influencing the logit difference score?

We can also start to see how our earlier observations on FOL, semantics, and pragmatics could potentially play out within the attention. Regarding FOL properties and relations, the model is attending to instances of `basket` where only `John` interacted with it. The model also shows attention patterns to instances of `basket` where it is the only entity `John` interacted with where the positive logit attribution score is high. This suggests that it could be representing a key relation between the subject, object and location anchored to specific instances of interaction.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/9a91c747-f3f6-47ad-8ecd-5124dbcbc79f"/>
<img src="https://github.com/user-attachments/assets/0492e03e-66de-49f3-af70-45918d8efc93"/>
<img src="https://github.com/user-attachments/assets/64a36cf9-5bc7-4212-ba60-08f08eb4a12a"/>
<img src="https://github.com/user-attachments/assets/f680eed9-8fe9-4636-9bd2-736f4a10424c"/>
</p>

<br>

Next, the fact that `the` attends to `basket` with a high positive logit attribution shows that the model could be inferring something about John’s awareness of the room's initial state, but also that he doesn’t have knowledge of what changed while he was away—this fits into a pragmatic understanding of the situation.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/579881a9-045c-4938-8fe7-4da932456770"/>
</p>

<br>

Semantically, when the model processes the second mention of `John` in the sentence, we can see that it’s attending `John` to every entity in the initial state. This could suggest the model is handling coreference resolution, linking `John` to the entities that were present at the start of the model's initial state of the scenario.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/fe668b4a-0b9f-411b-a1fd-93313c181c1c"/>
</p>

<br>

What’s really interesting is how the actions of `John`—both "taking" and "leaving"—impact the model’s attention to the entities that were mentioned initially. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/7379cba1-9af1-48d4-94bc-e3e271c2650b"/>
  <img src="https://github.com/user-attachments/assets/9d81a1aa-2a3e-4a26-8fea-26c02fde1f2e"/>
</p>

<br>

We can see how `from` in the phrase `...John comes back from...` attends to `school`. The model’s attention to `school` connects back to the earlier tokens that represent John’s actions and the initial state of the `room` before he left, suggesting how it could potentially integrate information across different parts of the sequence.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2a42142b-e522-4469-91b4-e8470dba85da"/>
</p>

<br>

It's entirely possible the model is balancing its handling of both semantic meaning and the logical relations between entities. 

We won’t dive into a full hypothesis about how the model works just yet—more on that later—but these are the kind of questions and iterative attention analysis that set the stage for figuring out the underlying circuit.

<br>


### ToM Circuit Discovery: Dictionary Learning, Sparse Autoencoders and Superposition

The linear representation hypothesis tells us that activations are **sparse**, **linear** combinations of **meaningful feature vectors**.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697598-f4d78700-4204-11eb-837c-40fb733f52df.gif" width="550px">
</p>

<br/>

This means there is some dictionary (data structure for storing a group of things) of concepts that the model knows about—what it's learned during training—and each one has a direction associated with it. On a given input some of these concepts are relevant, they get some score and its activations are roughly linear combinations of those directions weighted by how important they are eg. king is the male direction + the royalty direction. Sparsity comes into play because most concepts are not relevant to most inputs, eg. royalty is irrelevant to bananas, so most of the feature scores will be 0.

Sparse autoencoders (SAEs) are neural networks that learn both the dictionary and the sparse vector of coefficients. The key idea is to train a wide autoencoder to reconstruct the input activations so that the hidden state learns the coefficients of the meaningful combinations of neurons and the decoder matrix—the dictionary—learns the meaningful feature vectors and each latent variable in the autoencoder is a different learned concept.

The hope is that if there is an interpretable sparse decomposition—the output of the mechanism the autoencoder is learning from—it is now human interpretable.

This technique allows us to find abstract features that the model uses to represent concepts that the model uses to make predictions. These features are causually meaningful, and we can steer the model's output (behavior). So SAEs find real structure in the model that shows us how it is performing a task.

Even simpler, we can think of them as microscopes that lets us see inside language models to better understand how they work.

<br>

SAEs are based on the hypothesis that models have a big list of concepts they "know" about, w/ associated directions. On each input, only a few concepts matter and model internals are linear combinations of those directions. SAEs help find these directions (mention directions in residual stream that are read/written by attention and mlps). 

There are many directions to find because of 1) polysemanticity, where many neurons fire for multiple, often times unrelated features.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/068f4903-6b4f-4afe-9d8b-9ace61c16fc9" width="950px">
</p>

<br/>

And 2) superposition, so neural networks represent more concepts (features) than they have neurons and uses linear combinations of neurons to represent these concepts. 

Basically neurons represent multiple different things and features are spread across multiple different neurons. Because of superposition, we have a limited number of neurons for all our features, so there are lots of features and not so many neurons in any given activation space. But the irony is that the features are actually sparse, so only a few of them are active at any given time. This allows us to take advantage of SAEs. 

<br>

we relate the input to an intermediate value (SAE feature) or relate some intermediate values to the output

we can see how the model goes from simple to more complex features

It's purpose here is to help decode the ToM circuit.

(Figure out where this should go): SAEs give us a microscope that combats the curse of dimensionally and let’s us have a look inside of the internal mechanisms of transformers







<br>

(make sure I mention superposition briefly when introducing SAE representations of the ToM passage. Basically neurons represent multiple different things and features are spread across multiple different neurons.)

Because of superposition, we have a limited number of neurons for all our features, so lots of features and not so many neurons in any given activation space. But the irony is that the features are actually sparse, so only a few of them are active at any given time. This allows us to take advantage of SAEs. 

<br>



<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/e7869efd-bcaa-4d21-b49f-c0e1db1de148" width="480"/>
</p>


<br>

So we can take the activation vectors from attention, an mlp or the residual stream, expand them in a wider space using the SAE where each dimension is a new feature and the wider space will be sparse, which allows us to reconstruct the original activation vector from the wider sparse space, then we get complex features that the attention, mlp and residual stream have learned from the input. From this we can extract rich structures and representations that the model has learned and how it thinks about different features as its processing the input.  

<br>

The SAE suite used is Google Deepmind's <a href="https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/" title="Google Deepmind" rel="nofollow">Gemma Scope.</a> Its a collection of hundreds of SAEs on every layer and sublayer of Gemma 2 2B and 9B.




<br>

The features found here represents cases where the model learned about a specific behavior and it can then represent or replicate that feature. For example, if there were a secrecy feature that represents various ways in which you could be secretive -- black ops intelligence, keep secrets from friends etc -- you could increase that features activation and the model will plot about how it should keep things secret

What this shows us is that gradient descent --the optimization algorithm used to train modern language models-- is very smart and will learn things that we wouldn't even think to look for. SAEs are helpful here because we do not need to guess at what features gradient descent taught the model.

<br>

Using the trained SAE on the ToM passage (input to Gemma 2 2B), we can see which features in the model are activated.

<br>


### ToM Circuit Discovery: Activation Patching

Activation patching is a super useful technique that helps us track which layers and sequence positions in the residual stream are storing and processing the critical information we care about.

The obvious limitation of the techniques we’ve used so far is that they only focus on the final parts of the circuit—the bits that directly affect the logits. That’s useful, but clearly not enough to fully understand the whole circuit. What we really want is to figure out how everything composes together to produce the final output, and ideally, we’d like to build an end-to-end circuit that explains the entire behavior.

This is where activation patching comes in. First introduced in the ROME paper (where they called it causal tracing), activation patching lets us dig deeper into the model’s internal computations.

Here’s how it works: You run the model twice—once with a clean input that produces the correct answer, and once with a corrupted input that doesn’t. The trick is that during the corrupted run, you intervene by patching in an activation from the clean run. Basically, you replace the corrupted activation at a specific point with the corresponding clean activation and then let the model finish the run. The key insight here is that you can measure how much this patch shifts the output toward the correct answer.

By iterating over lots of different activations, you can map out which ones matter. If patching a certain activation makes a big difference in pushing the model toward the right answer, it tells us that activation is important for the task.

In other words, this is a noising algorithm (as opposed to the denoising focus we had in the last section).

The ability to localize computations like this is a huge win for mechanistic interpretability. If the model’s computations are spread out all over the place, it’s going to be much harder to form a clean, understandable story of what’s going on. But if we can pinpoint exactly which parts of the model matter, we can zoom in, figure out what they’re representing, how they’re connected, and ultimately reverse-engineer the circuit that’s driving the behavior.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/aa21ea4f-67e4-4ab6-a373-cac81c8a3ee5" width="700"/>
<br>
<small style="font-size: 8px;">Patching into a transformer can be done in a bunch of different ways (e.g. values of the residual stream, the MLP, or attention heads' output.). If you want to get really granular, you can patch at specific sequence positions (not shown). This flexibility lets us explore different components of the model and figure out exactly where certain behaviors are coming from.</a></small>
</p>

<br>

We can think of this activation patching algorithm as a form of noising, since we’re running the model on a clean input and introducing noise by patching in activations from the corrupted run. The flip side is denoising, where we start with a corrupted input and patch in activations from the clean input, effectively removing noise.

So, when would you use noising versus denoising? It really depends on your goals. Denoising typically gives you stronger results because demonstrating that a component (or set of components) is sufficient for a task is a big deal—it shows that this part of the model is doing something essential. But transformers are complex, and the components are deeply interdependent, so noising can sometimes lead to unpredictable outcomes. Just because performance drops when you ablate a component doesn’t automatically mean it was necessary for the task.

For example, if you ablate MLP0 in Gemma-2-2B, performance gets much worse across a bunch of tasks, but that doesn’t mean MLP0 is crucial for something like the ToM task. In fact, MLP0 seems to function more like an extended embedding layer—it’s generally useful for processing tokens but isn’t doing anything specific to ToM. We’ll dig deeper into this later, but the key point is that noising can lead to some ambiguous results, while denoising tends to give clearer answers.


<br>

### ToM Circuit Discovery: ToM Circuit

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/90a49f75-99a2-42ee-a619-5c9d4ec0d8a5" width="650"/>
</p>


<br>

<br>

# references:

Kosinski, *Evaluating Large Language Models in Theory of Mind Tasks.* Stanford University. 2023.[<a href="https://arxiv.org/pdf/2302.02083" title="Kosinski" rel="nofollow">1</a>]

Ullman, *Large Language Models Fail on Trivial Alterations to Theory-of-Mind Tasks.* Harvard. 2023.[<a href="https://arxiv.org/pdf/2302.02083" title="Ullman" rel="nofollow">2</a>]

Jamali, *Semantic encoding during language comprehension at single-cell resolution.* Nature. 2023.[<a href="https://www.nature.com/articles/s41586-024-07643-2" title="Jamali" rel="nofollow">3</a>]

Oguntola, *Deep Interpretable Models of Theory of Mind.*  Carnegie Mellon University. 2021.[<a href="https://www.nature.com/articles/s41586-024-07643-2" title="Oguntola" rel="nofollow">4</a>]

Le, *Revisiting the Evaluation of Theory of Mind through Question Answering.* Facebook AI Research. 2019.[<a href="https://aclanthology.org/D19-1598.pdf" title="Le" rel="nofollow">5</a>]

Ma, *Towards A Holistic Landscape of Situated Theory of Mind in Large Language Models.* University of Michigan. 2023.[<a href="https://arxiv.org/pdf/2310.19619" title="Ma" rel="nofollow">6</a>]

Jamali, *Unveiling theory of mind in large language models: A parallel tosingle neurons in the human brain.* Harvard. 2023.[<a href="https://arxiv.org/pdf/2309.01660" title="Jamali" rel="nofollow">7</a>]

Nguyen, *Language Models are Bounded Pragmatic Speakers: Understanding RLHF from a Bayesian Cognitive Modeling Perspective.* 2024.[<a href="https://arxiv.org/pdf/2305.17760" title="Nguyen" rel="nofollow">8</a>]

Wang, *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* Redwood Research, UC Berkley. 2022.[<a href="https://arxiv.org/pdf/2211.00593" title="Wang" rel="nofollow">9</a>] 

Htut, *Do Attention Heads in BERT Track Syntactic Dependencies?* NYU. 2019.[<a href="https://arxiv.org/pdf/1911.12246" title="Htut" rel="nofollow">10</a>]

Mikolov, *Linguistic Regularities in Continuous Space Word Representations.* Microsoft Research. 2013.[<a href="https://aclanthology.org/N13-1090.pdf" title="Mikolov" rel="nofollow">11</a>]

Yun, *Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors.* Facebook AI Research, UC Berkley, NYU. 2023.[<a href="https://arxiv.org/pdf/2103.15949" title="Yun" rel="nofollow">12</a>]

Riggs, *Really Strong Features Found in Residual Stream.* 2023.[<a href="https://www.lesswrong.com/posts/Q76CpqHeEMykKpFdB/really-strong-features-found-in-residual-stream" title="Riggs" rel="nofollow">13</a>]

Elhage, *A Mathematical Framework for Transformer Circuits* Anthropic. 2021.[<a href="https://transformer-circuits.pub/2021/framework/index.html#residual-comms/" title="Elhage" rel="nofollow">14</a>]

Bricken, *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* Anthropic. 2023.[<a href="https://transformer-circuits.pub/2023/monosemantic-features/index.html" title="Bricken" rel="nofollow">15</a>]

Cunningham, *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* EleutherAI, MATS, Bristol AI Safety Centre, Apollo Research. 2023.[<a href="https://arxiv.org/pdf/2309.08600" title="Cunningham" rel="nofollow">16</a>]

Templeton, *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Anthropic. 2024.[<a href="https://transformer-circuits.pub/2024/scaling-monosemanticity/" title="Templeton" rel="nofollow">17</a>]








