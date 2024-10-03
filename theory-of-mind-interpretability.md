# Theory of Mind and GPT models

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

<br>

## So What?

These concepts and processes can *help* explain how humans understand ToM, but are these concepts or processes mimicked in transformers? ToM prediction heavily relies on the context to make sense of the mental states and intentions behind the words, and the final word prediction is based on implied meanings and inferred intentions, which are central to pragmatics. Pragmatics encompasses understanding social interactions, cognitive states, understanding that others have mental states, beliefs, desires, intentions, and perspectives—that are different from one's own, which are key to ToM.

The remainder of this work will specifically focus on how GPT models will implement this task and in the end understand in a tractable way, the mechanisms responsible for completing the task across different heuristics and metrics.

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

It is a decoder-only transformer that has 25 layers and 7 attention heads per attention layer. The broader focus of this analysis is identifying the circuit that successfully models the ToM task, and the narrow focus is indentifying that circuit by understanding the behavior of the attention heads and MLPs.

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

### ToM Circuit Discovery: Identify Relavant Layers & Activations to the Task

Thanks to <a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" title="lesswrong.com" rel="nofollow">nostalgebraist</a> we have the logit-lens. So we can determine how language models refine their predictions across layers. The approach will be applied first to interpret layers and activations, and then to features and circuit discovery.

Causal interventions in the context of this analysis give way to techniques so that model components can be manipulated to understand or influence how different parts of the model contribute to the final output. In order to evaluate how model performance changes when performing causal interventions, we need a metric to measure model performance. 

The metric used here will be the logit difference, the difference in logit between the entity of the believed location of the object and the entity of the actual location of the object to gauge the accuracy of the models answers: `logit(basket) - logit(box)`.

We can use the same framework as the <a href="https://arxiv.org/pdf/2211.00593" title="Interpretability In The Wild: A Circuit For Indirect Object Identification In GPT-2 Small" rel="nofollow">Indirect Object Identification</a> (IOI) task as a basis for understanding ToM, because indirect object-subject entities can be mapped to original-new location entities.

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

What's interesting is that the model shows almost no capacity to handle the task until we get to layer 22. And then—boom—attention layer 22 kicks in and almost all the performance happens there, and then things get worse right after layer 23. It’s not just a smooth upward trajectory; there’s a clear peak followed by a clear descent.

So, what’s going on here? It’s a strong signal that layers 22, 23, and 24 are doing something really specific—writing to the residual stream in a way that allows the model to solve the ToM task. This insight can help us narrow the investigation and gives a clear direction: we need to figure out what kind of computation these layers are performing. It opens up exciting questions: How do attention layers (move information around) compare with MLPs (processes information) in their contribution? And within those attention layers, which heads are doing the heavy lifting?

This is where things get really fun. When narrowing down the problem, we can now start isolating the mechanisms and digging into specific computations, which will give real insights into how the model solves ToM.

Repeating the previous analysis, but for each layer by activation reveals how to begin the narrowing process.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/593f2793-f33a-4932-94be-d59a5d03a4d4" width="1700"/>
</p>

<br>

It looks like only the attention layers matter here. The ToM task, similar to the IOI task, is all about moving information around, pulling John's believed location of the cat into focus while ignoring the actual location of the cat. While there is minimal processing by the MLPs that matter (perhaps some level of understanding context is processed here), which warrents investivation, the emphasis is on the attention.

What’s particularly interesting is that attention layer 22 gives us a big boost in performance, but then things take a turn— MLP layer 22 and attention layer 23 and subsequent MLP layers actually make things worse. So, the attention mechanism is crucial, but there's a point where additional layers start to hurt more than help. This kind of dynamic tells us something important about how information flows through the model and where it can break down.

<br>

We can break down the output of each attention layer even further by looking at the sum of the outputs of each individual attention head. Every attention layer consists of 12 heads, and each head acts independently and additively to influence the final result.

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

### ToM Circuit Discovery: Attention Analysis

Attention heads are super valuable to study because we can directly analyze their attention patterns—basically, we can see which positions they pull information from and where they move it to. This is especially helpful in our case since we're focused on the logits, meaning we can just look at the attention patterns from the final token to understand their direct impact.

To help with this, I used the circuitsvis library to visualize these attention patterns. Specifically, we’ll be looking at the top 3 positive (visualizations for the negative heads were also produced in the analysis) based on their direct contribution to the logits.

One common mistake when interpreting attention patterns is to assume that the heads are paying attention to the token itself—maybe trying to account for its meaning or context. But really, all we know for sure is that attention heads move information from the residual stream at the position of that token. Especially in later layers, the residual stream might hold information that has nothing to do with the literal token at that position! For example, the period at the end of a sentence might store summary information for the entire sentence. So when a head attends to it, it’s likely moving that summary information, not caring if it ends with punctuation.

Understanding this distinction is key when studying how attention heads operate.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/527153fb-f75d-4ec2-a3e3-52a459740d41" width="1000"/>
</p>

<br>

Looking at this plot, it’s a good time to start thinking about the algorithm the model might be running. Specifically, for the attention heads with high positive attribution scores, we can see `the` is attending to `basket` with high confidence, particularly the second time basket is referenced, and `box` with lower confidence. How might this head’s behavior be influencing the logit difference score?

We won’t dive into a full hypothesis about how the model works just yet—that’s coming up after the next section—but this is the kind of question that sets the stage for figuring out the underlying mechanisms.

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
<small style="font-size: 10px;">Patching into a transformer can be done many different ways (e.g. values of the residual stream, the MLP, or attention heads' output.). We can also get even more granular by patching at particular sequence positions (not shown).</a></small>
</p>

<br>

We can think of this activation patching algorithm as a form of noising, since we’re running the model on a clean input and introducing noise by patching in activations from the corrupted run. The flip side is denoising, where we start with a corrupted input and patch in activations from the clean input, effectively removing noise.

So, when would you use noising versus denoising? It really depends on your goals. Denoising typically gives you stronger results because demonstrating that a component (or set of components) is sufficient for a task is a big deal—it shows that this part of the model is doing something essential. But transformers are complex, and the components are deeply interdependent, so noising can sometimes lead to unpredictable outcomes. Just because performance drops when you ablate a component doesn’t automatically mean it was necessary for the task.

For example, if you ablate MLP0 in Gemma-2-2B, performance gets much worse across a bunch of tasks, but that doesn’t mean MLP0 is crucial for something like the ToM task. In fact, MLP0 seems to function more like an extended embedding layer—it’s generally useful for processing tokens but isn’t doing anything specific to ToM. We’ll dig deeper into this later, but the key point is that noising can lead to some ambiguous results, while denoising tends to give clearer answers.


<br>
<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/90a49f75-99a2-42ee-a619-5c9d4ec0d8a5" width="650"/>
</p>


<br>






