# Deconstructing theory of mind in GPT models

<br>

#### Table of Contents <a id="top"></a>
- [Introduction](#introduction)
- [The relationship between theory of mind and language](#the-relationship-between-theory-of-mind-and-language)
- [So What?](#so-what)
- [Theory of Mind Circuit Discovery](#theory-of-mind-circuit-discovery)
    - [Principal Component Analysis](#principal-component-analysis)
    - [Identify Relevant Layers and Activations](#identify-relevant-layers-and-activations)
    - [Residual Stream and Multi-Head Attention](#residual-stream-and-multi-head-attention)
    - [Iterative Attention Head Analysis and Activation Patching](#iterative-attention-head-analysis-and-activation-patching)
    - [Dictionary Learning, Sparse Autoencoders and Superposition](#dictionary-learning-sparse-autoencoders-and-superposition)
    - [ToM Circuit](#tom-circuit)
    - [Copy Supressions role in the ToM Circuit](#copy-supressions-role-in-the-tom-circuit)
    - [Ablation Studies](#ablation-studies)
 - [Conclusion](#conclusion)
 - [References](#references)

 



<br>

*This post is a deep dive into the internals of transformer models. I'll assume you're comfortable with some basics, but I'll also be covering a lot of specific technical details along the way. Feel free to hop around using the contents—if you're already familiar with certain parts, you can jump straight to the results in the following sections<sub>[<a href="#iterative-attention-head-analysis-and-activation-patching" title="Go to section" rel="nofollow">1</a>]</sub><sub>[<a href="#tom-circuit" title="Go to section" rel="nofollow">2</a>]</sub><sub>[<a href="#conclusion" title="Go to section" rel="nofollow">3</a>]</sub>.*

<br>

# Introduction

<br>

<a href="https://arxiv.org/pdf/2407.02646" title="arxiv" rel="nofollow">Mechanistic interpretability</a> gives us a way to reverse engineer the internal workings of neural networks, turning the representations they learn into understandable algorithms. This helps us trace which parts of the model matter for a given task and decompose paths within the model into interpretable components.

With my current focus on transformer-based LLMs, theory of mind (ToM), and mechanistic interpretability, I've been wrestling with many core questions about ToM tasks:

How exactly do decoder-only language models (DOLMs) perform and *solve* ToM tasks? What's happening under the hood? What kinds of algorithms is the model relying on?

Is it appropriate to evaluate DOLMs the way a psychologist would analyze a human subject to gauge its level of ToM? One common framework for this is <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6974541/" title="ncbi.nlm.nih.gov" rel="nofollow">ATOMS</a> (Abilities in Theory of Mind Space), which categorizes concepts like beliefs, intentions, desires, emotions, knowledge, and percepts. Can we contextualize this behavior by zooming in and analyzing the internal mechanisms that enable ToM capabilities in these models? 

If a DOLM is trained across multiple ToM datasets representing different categories, and has robust performance across direct probing, and we find a clear algorithmic process —that leans heavily on the structure of language—to solve these tasks, does that automatically mean it's not really engaging in ToM, or could it be that this is the way models represent the abstract reasoning that ToM requires? 

Another key question is whether ToM tasks can be solved purely by leveraging linguistic properties and syntactic structures via compositionality. If functional compotence<sub>[<a href="https://arxiv.org/pdf/2301.06627" title="Mahowald" rel="nofollow">1</a>]</sub> (formal and social reasoning, world knowledge, situation modeling) can be achieved from exploiting linguistic signals that represent this compositionality, are these just "shortcuts" that "give answers away", or are they fundamental features that DOLMs rely on to perform and solve these tasks? 

I'm also asking myself: Do we even have a clear, interpretable algorithm for how *humans* solve ToM tasks? Outside of the scope of combining prior knowledge with observed behaviors and contextual nuances (intentionally ignoring emotions and cultural norms) in the human brain? 

Neural responses are dynamic and context-dependent, as seen in how the left prefrontal cortex encodes semantic information during speech processing. It suggests that the brain uses compositionality to process language<sub>[<a href="https://www.nature.com/articles/s41586-024-07643-2" title="Jamali" rel="nofollow">2</a>]</sub>, so maybe the way models handle this through linguistic structure isn’t that far off from certain aspects of human reasoning.

There’s always the argument that model brittleness is inevitable—no dataset, no matter how large, will cover every possible scenario. New, unseen ToM data could always "break" a model. But even beyond that, do the internal mechanisms for solving this problem remain consistent across different samples? While retraining on updated datasets could lead to short-term improvements, there’s still the broader challenge of evaluating the task effectively, given both our incomplete understanding of ToM and the limitations of DOLMs.

While I'm skeptical about why models are performing ToM or are not performing ToM, I think there’s value in breaking down the abstract reasoning involved in ToM tasks into interpretable algorithms or circuits. By understanding the internal representations in DOLMs, we can start to see how these models structure and approach ToM tasks—or more specifically false belief tasks. Even if they aren’t doing it like humans, we can still gain insights into the mechanisms they’ve learned for processing mental states.

<br>

# The relationship between theory of mind and language

<br>

In the human brain, the language network is a set of interconnected areas in the frontal and temporal lobes that handles everything from language comprehension to generation. It's highly tuned for various linguistic operations, covering everything from word meanings (semantics) to the broader context of conversations (pragmatics).

Humans have this amazing ability to infer the mental states of others using ToM. But conceptually, how could we represent ToM in a way that’s understandable for an algorithm? How might we frame it linguistically to help an algorithm get closer to understanding the mental states of others?

To explore how ToM could be represented algorithmically, let’s dig into a couple key linguistic principles: **Semantics**, and **Pragmatics**.

<br>

### Semantics

Semantics is all about representing meaning in language. It focuses on how words, phrases, and sentences convey meaning, and how humans interpret that meaning. It’s not just about the surface-level meaning of words, but also how those meanings combine and interact in context. Semantics covers a lot of ground, including things like compositional semantics, semantic similarity, word embeddings, distributional semantics, and distributed semantics.

For example, to linguistically understand the semantics of the ToM passage, we need to identify the entities, actions, relationships, and any implied meanings. To do this, we need to break down the sentence into all its entities and actions and map out how they interact. This is crucial for making sense of what's happening, especially when dealing with more abstract reasoning like ToM.

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

In semantics, understanding and interpreting this passage means extracting the meaning of each sentence, identifying entities, their properties, and relationships—core tasks of semantic parsing. Semantic parsing helps the model comprehend the context and infer implied meanings, which is essential for making accurate ToM predictions.

ToM involves representing complex mental states and expectations. For example, in this case, DOLMs can grasp both the underlying meaning and context, allowing them to predict that `John` thinks the `cat` is on the `basket`, even though it's actually on the `box`. This requires the model to go beyond the literal content, inferring the beliefs and mental states of the characters—key to performing these tasks.

<br>

### Pragmatics

A key concept in semantics, pragmatics focuses on how context influences the interpretation of meaning in language. This includes factors like speaker intent, conversational implicature, and situational context. To predict the final word in the example passage sequence, a model must understand not just the literal meaning of the words but also John's mental state, his expectations, and the context in which he is making the statement.

To obtain contextual understanding, we need to know the situational context—

`John` placed the `cat` on the `basket` before leaving for `school`, and he is unaware that `Mark` moved the `cat` to the `box` while he was away. 

Understanding John's beliefs and what he expects to find upon his return is crucial. We need the ability to infer the most likely location that fits John's expectation and the context (e.g., the `basket`). This involves recognizing that `John` thinks the `cat` is still where he left it, demonstrating the importance of pragmatics in interpreting language and predicting intended meaning.

<br>

## So What?
<sub>[Contents](#top)</sub>

<br>

These principles and operations can *help* interpret how humans perform ToM linguistically, but how do these concepts transfer to large language models in relation to ToM? 

By being trained for next word prediction, LLMs end up learning a lot about the structure of language, including linguistic features that were, until recently, thought to be out of reach for statistical models.

For example, a common way to test linguistic abstraction in LLMs is through probing. This involves training a classifier on internal model representations to predict abstract categories, like part-of-speech or dependency roles. The goal is to see whether these abstract categories can be recovered from the model’s internal states. Using this method, researchers have claimed that LLMs essentially "rediscover the classical NLP pipeline," learning linguistic features like part-of-speech tags, parse trees, and semantic roles across different layers.

ToM prediction heavily relies on context to make sense of the mental states and intentions behind the words and actions of others, and final word prediction is based on implied meanings (implicature) and inferred intentions (presupposition), which are central to pragmatics. Given the literature, even if the phenomena just statistical, **some** form of semantic and pragmatic inference in LLMs has been learned, regardless of how uneven or weak the performance.

<br>

# Theory of mind circuit discovery
<sub>[Contents](#top)</sub>

<br>

The broader goal of this analysis is to identify the circuit responsible for modeling the ToM task, with the more narrow focus being to pinpoint that circuit by understanding the behavior of attention heads, MLPs, and residual streams.

The model used for this analysis is Gemma-2-2B from Google's family of Gemma models. 

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

It is a decoder-only transformer that has 25 layers and 7 attention heads per attention layer.

In terms of the internal mechanisms of a language model, a **feature** is a property of the input that humans can understand and is represented in the model's activations (the tokens from the ToM passage). A **circuit** informs us of how these features are extracted from the input and then processed by the model to perform specific behaviors (e.g., reasoning), which gives us an algorithmic understanding of how the model works. So first, we analyze the features, use them to trace out circuits that connect and process those features, and once we understand more circuits we can better understand the model.

To look at ToM prediction through the lens of a decoder-only transformer, you can begin with a simplified, interpretable algorithm that focuses heavily on John’s mental state about where he placed the cat. This serves as a starting point to understand how the model might represent and process ToM-related reasoning: 

       - Consider events the subjects have witnessed.
       - Consider the location of objects based on the subject's last knowledge.
       - Ignore events that occurred while the subject was absent.
       - Predict subjects belief about the object's location based on the last event they witnessed.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2755018d-dd41-4bf7-adb7-d1f3ed087310" width="800"/>
<br>
</p>

<br>

### Principal component analysis
<sub>[Contents](#top)</sub>

<br>

Fitting PCA to the activations across MLP, attention, and residual stream patterns for the main entities, locations, and actions of the ToM passage reveals directions in the PCA space that show how the model is structuring the text internally. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/608b51e2-9ac2-4ced-a97c-9648d5d5cde0" width="950"/>
</p>

<br>

In the early MLP layers, clusters of tokens are starting to form. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/904aeee2-0796-4da9-8296-148071f46217" width="950"/>
</p>

<br>

In the middle layers, as the clusters become more distinct, they spread out. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/b3db3185-0cbc-4454-8460-dc5cf539f51d" width="950"/>
</p>

<br>

And the later layers show the most spread. Progressing through the layers, it seems tokens are clustering based on functional similarities in the text. Showing clear seperation of key tokens early on (John, cat, basket, box) and having close proximity to one another in later layers, showing what could be a false belief (John, cat, basket).

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/58e26654-1090-4b81-a21c-a2eedf450697" width="950"/>
</p>

<br>

The same can be said for the attention mechanisms, where in early layers distinct clusters emerge.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/add53282-d7dc-4522-a433-92efef5f8ade" width="950"/>
<img src="https://github.com/user-attachments/assets/3196c713-d8d0-4cec-b382-4a8025863452" width="950"/>
</p>

<br>

And in later layers, a refined focus in attention with more complex clustering patterns. Meanwhile, the residual stream seems to capture broader aspects of information, showing a more continuous evolution of representations across a wider context.

In the early attention layers, we’re seeing simpler, lower-level features, but as we move through the model, it’s clear the representations are getting more complex and structured. In later layers, the model appears to combine information from different parts of the input sequence, as shown by the mixed colors in various clusters. This likely reflects the temporal relationships between different elements of the sequence, and the positioning of key elements in these layers might represent the model's understanding of their roles in the narrative.

In the later layers, we’re picking up on some cool patterns: locations relevant to `John` and `Mark` seem to cluster, similar words in the attention heads are grouping up, and interestingly, `basket` is ranked higher than `box` in the residual stream hierarchy.

Even from this limited perspective, you can see how the model is capable of distinguishing concepts, integrating contextual information, and focusing on task-relevant features in each mechanism. The differences between each mechanism highlight how they contribute to this evolving representation. Attention heads seem especially important for forming distinct, task-relevant clusters of information in deeper layers, while the residual stream shows how information is continuously transformed as it flows between layers. And of course, pre- and post-processing in the residual stream gives us a view into how information gets reshaped before it moves to the next mechanism or layer. But more on that later.

<br>

### Identify relevant layers and activations
<sub>[Contents](#top)</sub>

<br>

Thanks to <a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" title="lesswrong.com" rel="nofollow">nostalgebraist</a> we have the logit-lens —so we can track how language models refine their predictions across layers. The approach will be applied first to interpret layers and activations, and then to dive deeper into feature and circuit discovery.

This technique is essentially a causal intervention—we're directly messing with parts of the model to figure out how they contribute to the output. Most of the methods in this analysis fit this kind of framework. 

To make sense of what’s happening, we also need a solid performance metric to track how things change when we intervene. That way, we can get a clear read on how the model's behavior shifts.

For the ToM task, where the goal is to distinguish between the believed and actual locations of objects, the model needs to predict both the original and updated locations after certain actions. The metric we’ll use here is logit difference, which represents the difference between the logit of the believed location and the logit of the actual location. In this case:
`logit(basket) - logit(box)`<sub>[<a href="https://arxiv.org/pdf/2211.00593" title="Wang" rel="nofollow">10</a>]</sub>.

When we deconstruct the residual stream using the logit-lens, we look at the residual stream after each layer and calculate the logit difference at that point. This simulates what would happen if we “deleted” all subsequent layers, giving us a snapshot of the model's evolving prediction. The final layernorm is applied to the residual stream values, which are then projected in the direction of the logit difference to measure the model's performance at each layer.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/40724e17-b54b-4d1b-aeff-0cdec72935a4" width="1700"/>
</p>

<br>

What's interesting is that the model shows almost no capacity to handle the task until we get to layer 22. And then—boom—attention layer 22 kicks in and almost all the performance happens there, and then things get a tiny bit better, then worse right after layer 24. It’s not just a smooth upward trajectory; there’s a clear peak followed by a clear descent after layer 24.

So, what’s going on here? It’s a strong signal that layers 22, 23, and 24 are doing something really specific—writing to the residual stream in a way that allows the model to solve the task. This insight can help us narrow the investigation and gives a clear direction: we need to figure out what kind of computation these layers are performing. It opens up exciting questions: How do attention layers (move information around) compare with MLPs (process information) in their contribution to this spike? And within those attention layers, which heads are doing the heavy lifting? What's going on in the residual stream exactly? What can we learn from the MLPs?

This is where things get really fun. When narrowing down the problem, we can now start isolating the mechanisms and digging into specific computations, which will give real insights into how the model performs ToM.

Repeating the previous analysis, but for each layer by activation reveals how to begin the narrowing process.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/593f2793-f33a-4932-94be-d59a5d03a4d4" width="1700"/>
</p>

<br>

Its clear that attention layers matter a lot. I'm not too surprised. I would imagine that the ToM task is centered around moving information around, pulling John's believed location of the cat into focus while ignoring or forgetting the actual location of the cat. While there is minimal processing by the MLPs that matter (perhaps some level of understanding context is processed here), which warrents investivation, the emphasis is on the attention.

What’s particularly interesting is that attention layer 22 gives us a big boost in performance, but then things take a turn—MLP layer 22 and attention layer 23 and subsequent MLP layers actually make things worse. So, the attention mechanism is crucial, but there's a point where additional layers start to hurt more than help. This kind of dynamic tells us something important about how information flows through the model and where it can break down.

We can break down the output of each attention layer even further by looking at the sum of the outputs of each individual attention head. Every attention layer consists of 7 heads, and each head acts independently and additively to influence the final result.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/3a62bf79-d9cc-4ccc-a455-fcf83bd86e04" width="650"/>
</p>

<br>

Interestingly, while there is positive activity that contributes to the prediction of the ToM task, only a few heads *really* matter. It seems many heads contribute—its possible that this distributed behavior is somehow important—but their activations appear quite weak. Head 3 at layer 0, head 4 at layer 22 and head 3 at layer 23 contribute positively on some range of significance, which kind of makes sense given the previously observed behavior on the attention in layer 22. On the flip side, head 7 at layer 18 and heads 5 and 4 at layers 23 and 25 respectively are negatively impacting the model greatly.

There are a couple of big meta-level takeaways here. First, even though our model has 7 attention heads in total, we can localize the behavior of the model to just a handful of key heads. This strongly supports the argument that attention heads are the right level of abstraction for understanding the model's behavior.

Second, the presence of negative heads is really surprising—like head 7 at layer 23, which makes the incorrect logit seven times more likely. I don’t fully understand what’s happening there, but the IOI paper touches on a few potential explanations. It's definitely something worth digging into more.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/6958f6c5-6b83-4337-ac3c-93baf669d565" width="950"/>
</p>

<br>

Looking back at the PCA output for layer 22, its clear that the model is doing something interesting in terms of concept clustering. The model is distinguishing between characters, objects and honing in on story elements that are crucial for ToM processing, but in a way where we can clearly see a refined heirarchical representation.

At this layer the residual stream pre is passing basic scence understanding to the attention. The attention is creating connections between tokens contextual relationships. The MLPs are computing new semantic relationships by processing objects in relation to their roles and actions. The residual stream post reconstructs the scene that encodes John's false belief and uncertainty with updated state information of spatial relationships clearly positioned relative to their connected objects. A clear decompose → relate → transform → reconstruct with updates pipeline.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/408baaa6-9596-41ac-94d1-098df04d129d" width="750"/>
</p>

<br>

Based on what I saw with PCA, I think its possible that the ToM task could be aligned with the linear representation hypothesis<sub>[<a href="https://arxiv.org/pdf/2311.03658" title="Park" rel="nofollow">11</a>]</sub><sub>[<a href="https://aclanthology.org/N13-1090.pdf" title="Mikolov" rel="nofollow">12</a>]</sub> –models seem to pick up properties of the input and represent them as directions in activation space. When we dig into layer 22's PCA, a few interesting things come to light.

The PCA breaks down into three clusters of concepts:

- Location tokens (`basket`, `box`, `room`)
- Actor tokens (`John`, `Mark`, `cat`)
- Mental state tokens (`thinks`, `knows`)

Looking at the residual stream post-PCA, we can see stronger associations between:

- `John` and `thinks`
- `basket` and initial state
- `box` and current state

When we compare the PCA with the linear plot above, its clear that the model is keeping two separate but parallel "tracks":

- Reality track (blue): represents actual events
- Belief track (red): represents John's belief state

The key thing here is that after Mark moves the cat, the two tracks split, but the belief track stays locked into John’s original understanding. This suggests that the model is able to simultaneously track reality and belief simultaneously, keeping them separate but interrelated to maintain parallel states. Even as the sequence progresses—Mark and John’s actions, them leaving, returning—the belief state remains consistent.

What’s also cool is that the PCA related tokens spatially, keeping clear distances between the conceptual groups. So, there’s this clear linearity across time, the temporal sequence, belief state maintenance, subject-action links, and object-location associations.

<br>

### Residual stream and multi-head attention
<sub>[Contents](#top)</sub>

<br>

Attention heads are valuable to study because we can directly analyze their attention patterns—basically, we can see which positions they pull information from and where they move it to. This is especially helpful in our case since we're focused on the logits, meaning we can just look at the attention patterns from the final token to understand their direct impact.

One common mistake when interpreting attention patterns is to assume that the heads are paying attention to the token itself—maybe trying to account for its meaning or context. But really, all we know for sure is that attention heads move information from the residual stream at the position of that token. Especially in later layers, the residual stream might hold information that has nothing to do with the literal token at that position! For example, the period at the end of a sentence might store summary information for the entire sentence up to that point. So when a head attends to it, it’s likely moving that summary information, not caring if it ends with punctuation. This makes it hard to asses what the attention heads are doing when tokens are being attended to. 

But at the same time, I think when an attention head is attending to a token, it is accessing abstract information stored at that position.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/c64273c7-5d0b-4efc-bbd1-b0ed05842aa5" width="280"/>
</p>

<br>

In transformer architectures, each token position has a residual stream—a vector that carries forward information as the model processes each layer. We can think of the residual stream as the place where everything communicated from earlier layers are communicated to later layers. It captures everything the model has *thought* so far, so it will contain everything important going on in the model.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/d1590634-0cb0-42f5-b177-a17ee0203af1" width="280"/>
</p>

<br>

The residual stream in a transformer isn’t just about processing token embeddings; it’s an information highway that aggregates outputs from previous attention heads and MLPs. Both attention heads and MLPs read from this stream, apply their edits, and then write the modified info back into the residual stream using linear operations (just simple addition). This linearity is key—it allows the input to any layer be decomposed as the sum of contributions from various mechanisms across different layers.

By the later layers, the residual stream holds rich, high-level abstractions: syntactic structures, semantic relationships, and even summaries of phrases or entire sentences. This enables the model to map syntax onto semantics in a powerful way. Attention heads read from specific positions in the residual stream and write new information to target positions, which helps move abstract, context-heavy information around—independent of specific tokens.

Going back to our period example, at the position of a period at the end of a sentence, the residual stream might hold a summary of the entire sentence rather than just the token embedding for the period itself. This layered representation is built up across attention blocks and MLPs, incorporating syntactic roles, semantic meanings, and sentence structure. Attention patterns help transfer these complex, high-level abstractions between positions, enabling the model to handle hierarchical structures.

As the model processes information, each layer can access everything from the residual stream **but focuses on specific directions** that are relevant for the task based on the similarity of information held between mechanisms. After aligning with the directions it needs, the model writes the information to another mechanism. The flow of information between mechanisms depends on how similar the directions in the residual stream are, guiding the movement of abstract information across the model. 

More on how transformers process information using linear algebra <a href="https://youtu.be/wjZofJX0v4M?si=yzNyY0gmwQ892Z6P&t=747" title="3Blue1Brown" rel="nofollow">here.</a>

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/61e58090-4367-4a9d-b8bb-411fcb5f0e1b" width="280"/>
</p>

<br>

Rather than the input needing to go through every single layer of the network, the model can choose which layers it wants information to go through via the residual stream and what paths it wants to send information to. This is why we can expect model behavior to be kind of localized, so as the input goes through each mechanism, not every piece of the input will receive an activation.

The model is using the residual stream to achieve compositionality between different pieces of information, and its how mechanisms in the model communicate with each other. 

For example, there could be some attention head in layer 2 that composes with some head in layer 22. Technically this looks like some head in the 1st layer will output some vector to the residual stream, the head in the 2nd layer will take as an input the entire residual stream and mostly focus on the output of the 1st layer and run some computation on it. For any pair of composing pieces in the model, they are completely free to choose their own interpretation of the input, so there's no reason that the encoding of the information between head 0 in layer 0 and head 5 in layer 3 will be the same as the encoding between head 2 in layer 0 and head 3 in layer 1. This means we can expect the residual stream to be very difficult to interpret.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/415c58d0-6975-4483-808c-31cccf887cd9" width="7500"/>  
</p>

<br>

So, what’s happening here is the model builds up hierarchical representations of language—phrases within sentences, sentences within paragraphs—and tracks sequences of events, which is particularly important for tasks like ToM, where understanding the events, the order of events, character actions and possibly even directional or spatial information is key.  In this framework, attention heads work like routers, directing specific pieces of information to the right places to solve the task. They aren’t just focusing on literal tokens but transferring abstract concepts like *"the last place John saw the cat"*, which aren't tied to any single token but are encoded in the residual stream.

This kind of hierarchical, nested structure in the residual stream is key to solving the ToM task. It requires the model to track what each character knows or believes over time, which means keeping updated representations of these abstract knowledge states in the residual stream.

In any case, it’s easy to get tricked if you think an attention head is just focusing on a literal token. We should be looking at this information alongside the information stored in the residual streams at that position—which often contains abstract concepts or higher-level representations.

While keeping all of that in mind, when looking at the plots, it’s a good time to start thinking about the algorithms the model might be using. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/527153fb-f75d-4ec2-a3e3-52a459740d41" width="1000"/>
    <small style="font-size: 8px;">The attention patterns of the heads. We can see where each token attends by the maximum value of where its attending.</a></small>
</p>

<br>

Specifically, for the attention heads with high positive attribution scores, we can see `the` is attending to `basket` with high confidence, particularly the second time basket is referenced, and `box` with lower confidence. How might this head’s behavior be influencing the logit difference score?

We can start to connect some dots between our earlier observations on semantics and pragmatics, and how they might show up in the model's attention patterns. We see that the model’s attention focuses on specific instances of the `basket`, especially when `John` is the only one interacting with it. This hints at the model potentially locking onto a key relation—between the subject `John`, the object `basket`, and the location—tied to those specific interaction moments.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/9a91c747-f3f6-47ad-8ecd-5124dbcbc79f"/>
<img src="https://github.com/user-attachments/assets/0492e03e-66de-49f3-af70-45918d8efc93"/>
<img src="https://github.com/user-attachments/assets/64a36cf9-5bc7-4212-ba60-08f08eb4a12a"/>
<img src="https://github.com/user-attachments/assets/f680eed9-8fe9-4636-9bd2-736f4a10424c"/>
</p>

<br>

This attention pattern suggests the model is encoding subject-object-location agreement and becoming more prominent in cases where the interaction is clear and exclusive to John.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/579881a9-045c-4938-8fe7-4da932456770"/>
</p>

<br>

The fact that `the` attends to `basket` with a high positive logit attribution in relation to other positive attributions at different positions in the passage is pretty telling. It suggests that the model is inferring John’s awareness of the location of the `cat`, specifically that it's on the basket. But at the same time, it seems like the model recognizes that John lacks knowledge about any changes that happened while he was away. This fits nicely into a pragmatic understanding of the situation—John’s belief is anchored to the initial state, and the model ignores what he does not know.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/fe668b4a-0b9f-411b-a1fd-93313c181c1c"/>
</p>

<br>

Semantically, when the model processes the second mention of `John` in the sentence, it’s throwing attention to every entity that was part of the initial state. This looks a lot like coreference resolution in action—linking `John` back to the same entities he was connected to at the start of the scenario. Basically, the model’s tracking `John` across mentions and making sure it’s keeping all the initial context straight. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/7379cba1-9af1-48d4-94bc-e3e271c2650b"/>
  <img src="https://github.com/user-attachments/assets/9d81a1aa-2a3e-4a26-8fea-26c02fde1f2e"/>
</p>

<br>

What’s really interesting is how the actions of `John`—both "taking" and "leaving"—impact the model’s attention to the entities that were mentioned initially.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2a42142b-e522-4469-91b4-e8470dba85da"/>
</p>

<br>

We can see how `from` in the phrase `...John comes back from...` attends to `school`. The model’s attention to `school` connects back to the earlier tokens that represent John’s actions and the initial state of the `room` before he left, suggesting how it could potentially integrate information across different parts of the sequence.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/a1ff26f7-85ff-42f5-8cbf-7028cd6ff6a5"/>
</p>

<br>

We can also color everything from source token to output token to see how much every token effects every other token. These are all the tokens that have their probabilities increased by the attention heads. We can see that `the` attended back to `box`, `basket`, `cat` which increased the probability that the next token should be `room`, suggesting noun phrases and more complex compositional patterns in the future.

We won’t dive into a full hypothesis about how the model works just yet—more on that later—but these are the kind of questions and iterative attention analysis that set the stage for figuring out the underlying circuit.

<br>

### Iterative attention head analysis and activation patching <a id="iterative-attention-head-analysis-and-activation-patching"></a>
<sub>[Contents](#top)</sub>

<br>

To trace which parts of the model's attention are key for this task, and break down those pathways, we need a deeper dive into the attention patterns. Specifically, we want to see how the model attends to tokens related to John, his initial actions, and his final actions.

One approach is tracking the activations of key tokens (John, basket, box, cat) across layers, showing how their representations evolve. Another approach is pinpointing which attention heads contribute most to predicting "basket."

By combining these methods and comparing the results, we can zero in on heads that attend to both the initial state and John’s final action.

Looking at the most basic units of computation in the attentions heads will give the most fine-grained account of what is happening when the model is processing information to be sent to the MLPs. So we need to explore the roles of the query (Q), key (K), and value (V) vectors across the hierarchy of layers.

The DOLMs attention mechanisms weigh the importance of different parts of the ToM passage. Each attention head computes three components:

- **Query (Q):** Determines which token positions to attend to.
- **Key (K):** Represents the tokens considered for attention at each position.
- **Value (V):** Contains the information to be propagated forward.

The way QKV attention works is sort of like how a search engine operates. Imagine you’re looking for a video on YouTube —the text you type in the search bar is your query. The search engine then compares that query to a bunch of keys —like video titles, descriptions, tags that are stored in its database. Finally, it retrieves and ranks the best-matching videos —which are the values.

So, attention is basically about mapping a query to the most relevant keys and pulling out the corresponding values.

In somewhat technical terms, the values for the QK vectors control how much attention each token pays to others within the attention mechanism. A larger Q relative to K suggests the current token is more strongly driving the attention, meaning it's "searching" for relevant information to attend to. On the other hand, when K is larger than Q, it indicates that the token associated with K is drawing more attention from other tokens—essentially, it's being "attended to." The Vs hold the actual information or features from the input tokens and play a crucial role in determining what information is passed forward once the attention scores between Q and K are calculated.

However, it's important to note that the relative sizes of Q and K don't directly determine who is "doing the attending." Instead, both vectors interact through dot-product attention: Q represents the token initiating the attention (the one trying to find relevant content), and K represents the token being attended to (the potential source of relevant information). The attention scores are computed based on the interaction between Q and K, meaning both vectors play a role in deciding where attention is focused. The difference in their values might offer clues about the roles of specific tokens in the attention process, but both vectors contribute to the overall mechanism.

Selecting a few heads across layers, we can see how things are playing out in the context of the last token `basket` being predicted.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2f657659-d0eb-4385-98e1-fb0b984441b5" width="700"/>
<img src="https://github.com/user-attachments/assets/c4909dbf-f729-458c-9de9-b742cfdbffbf" width="700"/>
<img src="https://github.com/user-attachments/assets/d801aeb9-36f4-4ef4-95ae-958a80081bc6" width="700"/>
<br>
</p>

<br>

- **Early Layers & Heads 0-10:**
    - **Layer 0, Head 7:** Attends to elements like `in` and `on` and other adpositions
    - **L5, H2:** Attends to `basket` and `box`, punctuation and the beginning of the sequence
    - **L8, H0:** Shows signs of growing attention to article-noun agreement `the cat`, `the box`, `the basket`, `the room`  
    - **L10, H0:** Strong focus on `is`, `on`, and `cat`, consolidating scene representation
    - **L10, H1:** High attention to `on` and `cat`, primarily focused on retrieving information rather than combining, Q vector spikes for subject-verb agreement `John takes`, `Mark takes`, as well as consistent attention to main verbs with minimal K activations
      <br>
    - **L10, H4:** Begins to differentiate between `box` and `basket` in a specialized way via prepositional phrases—`on the basket`, `off the basket`, with high activation on `the` in the last position of the sequence, indicating learned spatial relationships. Compared to head 1 in the same layer, strong V spikes for verb-object agreement (`takes the cat`, `puts it`). Highest v spikes around complete action sequences (`takes the cat and puts it on`)

- **Middle Layers & Heads 10-17:**
    - **L14, H0:** Attention to `basket`, `box`, and `cat`, showing clear object differentiation, increased attention to `basket`, starting to discover belief state
    - **L14, H3:** Very high attention to `box`, possibly encoding the actual state
    - **L14, H6:** Increasing attention to `basket` compared to `box`, suggesting comparison
    - **L16, H0:** Focuses on `room` with moderate attention to objects and spatial relationships
    - **L16, H2:** Strongly attends to `box`, `basket`, and `cat`, refining object relationships
    - **L16, H3:** High attention to `cat` and `on`, objects and spatial relationships
    - **L16, H7:** Very strong, specialized attention to `box` and `basket`, and possibly comparing locations
    - **L17, H0:** Strong attention to `box` in its 2nd position in the sequence and `basket` at its 2nd position in the sequence, beginning of sequence -maintaining scenario context.
    - **L17, H3:** Very high attention to `box`, reinforcing possibly reinforcing where the cat is actually located
    - **L17, H4:** Increased focus on `basket`, and determiners beginning to emphasize the belief state
    - **L17, H6:** Attends to mainly determiners, especially the final one at the end of the sequence
    - **L17, H7:** Extremely high attention to `on`, `is`, `off` solidifying spatial relationship encoding via adpositions

<br>

We can see the model building its representation across layers, with later layers showing stronger activations for key tokens. Early to middle encodings suggest relations between grammar, spatial relationships, and initial object-subject integration. The middle to late encodings seem to refine object representations, begin to emphasize John's belief state and then strongly maintain that state.

We can sort of see evidence for copying heads (attend to a token and increase the probability of that token occuring again) in layer 0 head 7 and layer 10 head 1. Both showing rigid, position-based patterns, clean isolated spikes. The former shows strong Q spikes at regular intervals with minimal KV interference, it seems to be doing token-level copying or positional tracking, but the sharp, forward, diagnoal increased magnitude of Q spikes screams systematic copying with position awareness to me. The latter shows copy-like behavior for specific syntactic structures with regular patterns around sentence boundaries and copying verb-related information forward.

Evidence for <a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" title="Olsson" rel="nofollow">induction heads</a> (look at present token in context, look back at similar things that have happened, predicts what will happen next) in layer 14 head 0 and layer 17 head 3. Both showing more flexible semantic-based patterns, and sharp, backwards K spikes and slight sharp forwards Q spikes. The former shows strong QK spikes at semantically similar tokens, attention to repeated patterns of actions/states, and the latter showing the tracking of recurring patterns in character actions, and next state predictions based on previous patterns.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/43905290-8648-435d-820a-9526d971fe0a" width="700"/>
<br>
</p>

<br>

Specifically, for the asymmetric patterns in layer 22 head 4, the highest Q attention (blue spike) is at the beginning of the sequence, around `basket` in the first mention of the basket, maybe suggesting the model is strongly querying the initial state of where the cat was placed (might be an artifact given its almost everywhere). The V attention (green) show strong contributions around `basket` early in the sequence, completely dominating the V attention of `box`, and several medium-height spikes around key events in the story (like when the cat is moved).

The pattern shows the model is attending strongly to both the initial state (`cat on basket`) and the intermediate state (`cat moved to box`). The high query attention to the initial `basket` placement suggests the model understands this is relevant to John's belief state, and even captures `John` in the initial state with high attention activations relative to `Mark`. The value contributions from both `basket` and `box` mentions show the model is tracking both possible locations of the cat. It's tracking both the real state (`cat on box`) and John's believed state (`cat on basket`). 

The strong attention to the initial state makes sense since that's what John last saw before leaving. The model also appears to be using this head to integrate information about object locations and character knowledge states. This head is likely key in some belief state emphasis context, and likely follows a collection of heads that build up to this attending to John's false belief. 

<br>

More formally, for each token position we have QKV vectors, 

<code>Q<sub>i</sub></code> <code>K<sub>i</sub></code> <code>V<sub>i</sub></code>

<br>

And the attention score for the tokens position to another positions,

score(<code>i,j</code>) = softmax((<code>Q<sub>i</sub></code> · <code>K<sub>j</sub></code>) / √<code>d<sub>k</sub></code>)

<br>

And output for position `i` is,

out<code><sub>i</sub></code> = Σ<sub>j</sub>(score(<code>i,j</code>) × <code>V<sub>j</sub></code>

<br>

For the 4th head of layer 22 , the QKV vectors for the attention mechanism will look something like this,

<code>Q<sub>basket</sub></code> ≈ 1.0 (tall blue spike)
<code>K<sub>basket</sub></code> ≈ 0.3 (red line)
<code>V<sub>basket</sub></code> ≈ 0.8 (tall green spike)

<code>K<sub>box</sub></code> ≈ 0.2 (red line)
<code>V<sub>box</sub></code> ≈ 0.4 (medium green spike)

score(<code>basket,basket</code>) = softmax((<code>Q<sub>basket</sub></code> · <code>K<sub>basket</sub></code>) / √<code>d<sub>k</sub></code>)
≈ softmax((1.0 × 0.3) / √64)

score(<code>basket,box</code>) = softmax((<code>Q<sub>basket</sub></code> · <code>K<sub>box</sub></code>) / √<code>d<sub>k</sub></code>)
≈ softmax((1.0 × 0.2) / √64)

out<code><sub>basket</sub></code> = score(<code>basket,basket</code>) × <code>V<sub>basket</sub></code> + score(<code>basket,box</code>) × <code>V<sub>box</sub></code>

out<code><sub>basket</sub></code> = score(<code>basket,basket</code>) × <code>V<sub>basket</sub></code> + score(<code>basket,box</code>) × <code>V<sub>box</sub></code>

<br>

Where the tall blue spike for `basket` is implemented via the strong Q vector weighting, which helps the model search for or focus on John's initial belief state. 

The strong green spikes for both `basket` and `box` positions V vectors carries location information. 

The moderate red activity combines both states, weighted by attention scores, allowing the model to maintain a strong representation of John's initial belief state of the `basket` location (false belief, contradiction), track current state of the `box` location (true belief, reality), and weight them appropriately for belief state tracking.

In terms of linguistic representations, notice the attention patterns around `He` - there are small but noticeable spikes in both Q and V contributions when pronouns need to be resolved back to their referents (`John`, `Mark`). 

There are signs of temporal sequence markers, attention spikes around temporal transition phrases like `while` and `when`, helping track the sequence of events and time periods (before/during/after John's absence). 

Attention patterns showing action-state-verb agreements, tracking state changes through verbs. Small but consistent attention to prepositions like "on" and "off" that describe spatial relationships, which work together with the objects (basket/box) to establish location states. And there's attention around verbs that relate to mental states like "knows" and "thinks", marking belief states.

Overall it appears by this layer the model has integrated information from earlier layers and focuses on more complex contextual/semantic relationships!

This output is suggesting that the model is composing features related to objects and their locations, with a strong focus on `basket` in the final layers. The token `basket` shows a significant increase in activation from layer 22 onwards, maintaining high activation through the final layer. This suggests that the model is maintaining the information about the initial state (`cat` on `basket`) despite contradictory information introduced later in the passage. Tracking where `John` thinks the `cat` is seems to be the most important feature of this head.

In relation to this, we can also see the suppression of the actual current state (`cat on box`) in favor of the believed state (`cat on basket`). This suppression head seems to primarily operate in layers 23 and 25, heads 5 and 4 playing a crucial role. So this head maintains the activation of `basket` while relatively suppressing `box`, which would be preserving John's false belief about the cat's location. This can be observed in several ways:

**Attention patterns:**

- Many heads in layers 22-25 show high attention to `basket` and relatively lower attention to `box`.
- Layer 23, head 5 and head 6 show particularly strong attention to `basket` over all instances of the token in the sequence, where `box` activations are relatively low.

**Activation patterns:**

- In the final layers (22-25), `basket` consistently has higher activation than `box`, despite `box` being the actual current location of the `cat`.

<br>

Activation patching is a super useful technique that can help us track which layers and sequence positions in the residual stream are storing and processing the critical information we're interested in.

The obvious limitation of the techniques we’ve used so far is that they only focus on the final parts of the circuit—the bits that directly affect the logits. That’s useful, but clearly not enough to fully understand the whole circuit. What we really want is to figure out how everything composes together to produce the final output, and ideally, we’d like to build an end-to-end circuit that explains the entire behavior.

This is where activation patching comes in. First introduced in the ROME paper (where they called it *causal tracing*), activation patching lets us dig deeper into the model’s internal computations.

Here’s how it works: You run the model twice—once with a *clean* input that produces the correct answer, and once with a *corrupted* input that doesn’t. The trick is that during the corrupted run, you intervene by patching in an activation from the clean run at a specific point in the network. Basically, you replace the corrupted activation at a certain layer and position with the corresponding clean activation and then let the model continue its computation. The key insight here is that you can measure how much this patch shifts the output toward the correct answer, we can assess the importance of that particular activation.

By iterating over lots of different activations, you can map out which ones matter. If patching a certain activation makes a big difference in pushing the model toward the right answer, it tells us that activation is important for the task.

In other words, activation patching functions as a noising algorithm, contrasting with the denoising approaches we've previously focused on.

The ability to localize computations like this is a huge win for mechanistic interpretability. If the model’s computations are spread out all over the place, it’s going to be much harder to form a clean, understandable story of what’s going on. But if we can pinpoint exactly which parts of the model matter, we can zoom in, figure out what they’re representing, how they’re connected, and ultimately reverse-engineer the circuit responsible for the observed behavior.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/aa21ea4f-67e4-4ab6-a373-cac81c8a3ee5" width="700"/>
<br>
<small style="font-size: 8px;">Patching into a transformer can be done in a bunch of different ways (e.g. values of the residual stream, the MLP, or attention heads' output.). If you want to get really granular, you can patch at specific sequence positions (not shown). This flexibility lets us explore different components of the model and figure out exactly where certain behaviors are coming from.</a></small>
</p>

<br>

We can think of activation patching as a form of noising. In this approach, we run the model on a clean input but introduce "noise" by patching in activations from the corrupted run. The flip side is denoising, where we start with a corrupted input and patch in activations from the clean run, effectively removing noise.

So, when would you use noising versus denoising? It depends on your goals. Denoising typically gives you stronger results because demonstrating that a component (or set of components) is sufficient for a task is a big deal—it shows that this part of the model is doing something essential. But transformers are complex, and the components are deeply interdependent, so noising can sometimes lead to unpredictable outcomes. Just because performance drops when you ablate a component doesn’t automatically mean it was necessary for the task.

Take MLP0 in Gemma-2-2B in the logit difference from each head plot above for instance. If you ablate it, performance gets much worse across a bunch of tasks, but that doesn’t mean MLP0 is crucial for something like the ToM task. In fact, MLP0 seems to function more like an extended embedding layer—useful for processing tokens but isn’t doing anything specific to ToM. We’ll dig deeper into this later, but the key point is that noising can lead to some ambiguous results, while denoising tends to give clearer answers.

**Activation Patching:**

- Layer 22, head 4 shows a large positive logit difference, indicating that this head is crucial for the final prediction of `basket`.
- This suggests that layer 22, head 4 could be a key component of a suppression circuit, focusing on maintaining the believed state.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/e9680ec6-8c8e-4afe-90a6-0b20c59c53d4" width="500">
</p>

<br/>

The model seems to start by encoding initial information about objects and characters in the early layers. As we move into the middle layers, it builds up a more detailed representation of the scene and the actions taking place. By the later layers, particularly from layer 22 onwards, it focuses heavily on the believed state of the world (e.g., `the cat on the basket`).

An important thing to note is that these functions are not neatly isolated but distributed and overlapping across multiple attention heads. For instance, several heads work together to represent the "mental state," and many of these heads also contribute to other tasks. The suppression activity, for example, doesn’t come from a single head—it emerges from the interactions between multiple heads in the later layers.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/a424bb3e-90f7-4992-ab3f-3fd26ba45ebe" width="900">
  <img src = "https://github.com/user-attachments/assets/1670b458-7601-4449-b901-3df3e706dfac" width="900">
</p>

<br/>

Diving deeper into the activation patching results, focusing on the residual stream/midstream, blue regions indicate where patching helped the model get closer to the correct prediction (basket), while red regions show where patching hurt (pushing it towards box). The clean run is the uncorrupted input—where the model gets things right ("John thinks the cat is on the box"). The corrupted run comes from swapping adjacent tokens, which messes up the sentence’s meaning and leads to wrong answers. The goal is to patch activations from the clean run into the corrupted one at various layers and sequence positions and see how much it improves the model’s logit difference (i.e., how much closer it gets to predicting the correct answer).

Patching the `box` token at layer 1 gives a massive boost, almost recovering full performance. But, as we move to later layers, the **most impactful patching** happens at the final `the` token before the blank. **This shift hints at something important:** the model first focuses on where the `cat` was (`on the box`), and later on, it shifts to what word needs to be filled in (`basket` vs. `box`). There’s a super interesting pattern—starting from the `box` token in layer 0 and running up to the final `the` token in layer 25. This implies a distinct computational flow across the model’s layers. Early on, (layers 0-5) it’s all about the `box` token (likely where the model locks in the idea that the cat was on the box).

 Between layers 5-20, the patching impact spreads more evenly across tokens. This is probably where the model’s pulling everything together, building up a complete understanding of what’s going on. Then, by layers 20-25, the focus shifts hard onto the final `the` token—this is where the model's deciding which word (`basket` vs. `box`) to predict. While patching `box` is super helpful in early layers, it starts to hurt later on (negative blue regions). It seems like **the model needs to remember the original cat position** (`box`) early on but **then "forget" it** by the end to make the right call (`basket`). This shows how the model's thinking evolves layer by layer.

One cool takeaway is how localized the effect is—patching just a few tokens or layers can fix a lot of the model’s mistakes. It’s not spreading out the info evenly across the whole network. Instead, there’s a very directed flow of information from `box` to `the` over time.

The midstream plot looks similar but now we’re patching activations midstream. Here, the biggest improvements show up in layers 20+, where the logit difference for the `box` drops more. The overall patterns match the first plot, but the intensity is lower, suggesting that patching mid-layer activations has a more diffuse effect compared to early residual streams.

**This fits with the bigger picture:** earlier layers are encoding the critical scene details (e.g., Mark moving the cat), while midstream activations are key for representing changes in location (whether the cat ends up on the basket or box).

The whole process aligns with previous attention analysis—early layers set up the scene, mid layers handle object movement and maintaining the scene, and late layers focus on resolving John’s false belief.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/b4d4f65b-6628-42a3-bb4b-de89915c8b82" width="950">
</p>

<br/>

This plot shows the effect of patching attention head outputs at each layer and head. The color shows how much patching that specific head shifts the model's prediction from `box` to `basket`.

The biggest takeaway? The last few layers, especially layers 21 through 25, matter the most. Specifically, layer 25 drives a huge shift towards the correct prediction (`basket`). This fits the pattern we’d expect—later layers are where the model locks in its final decision. Small tweaks to attention outputs here can dramatically change the model’s output, whereas earlier layers are more about building up representations.

What’s really interesting is that the important heads in layer 25 weren’t necessarily important in earlier layers. This suggests that the role of each head evolves over time—it’s not just a linear transformation from layer to layer. Instead, heads are integrating new information from other heads and the residual stream in complex ways. 

At most layers, only a handful of heads have a significant effect when patched. Most heads stay neutral (near white), meaning patching them doesn’t really change the output. So, the computation relies on a sparse set of heads, not an even distribution of information across all heads.

It’s interesting to compare this to the attention breakdown by query, key, value. For example, head 5 at layer 20 mattered in both the query and output, but head 6 at layer 25 is only critical in the output—not the query, key, or value. This suggests the queries it’s attending to in layer 25 don’t need to be super precise, but the values it outputs are crucial for driving the final decision.

Just a few heads at a few layers carry most of the critical information needed for the model’s final prediction. But these heads take on different roles at different layers, and their importance can shift dramatically. The model is clearly doing a complex, multi-step computation—transforming representations layer by layer to reach the right conclusion.

The residual stream shows strong activity in the early layers, indicating the importance of initial context.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/cf67fbe1-e894-4118-9473-52c98f41d881" width="1000">
</p>

<br/>

This plot gives us a deep dive into what’s happening inside attention heads. Each attention head does two key things: **1)** deciding where to move information (this is governed by the attention pattern, which the QK circuit handles) and **2)** deciding what information to move (handled by the value vectors, controlled by the OV circuit). To figure out which part matters more, we can patch just the attention pattern or the value vectors separately.

Let’s start with the 'z' plot (the head's output). Patching outputs from certain heads has a noticeable effect, shifting the model’s output from “box” to “basket,” particularly in the last 5-10 layers. Specifically, head 2 in layer 20 and head 5 in layers 15 and 20 have the largest impact.

Now, looking at the 'q' plot (query vectors), we see a similar pattern. Heads like head 4 in layers 15 and 20 stand out. This suggests that **adjusting which queries these heads focus on** is pretty important for guiding the model toward the correct outcome. The strongest signals here are in the early to mid layers, where the model is starting to grapple with the false belief.

The 'k' plot (keys) is less clear, though head 4 at layer 15 and head 3 at layer 20 still seem to matter. The queries and keys work together to determine which parts of the input each head attends to, so it’s not surprising that they both play a role. **Comparing query vectors** (which represent John’s perspective) **with key vectors** (which capture what’s actually happening) **helps the model reconcile John’s false belief**. The value vectors, on the other hand, carry the ground truth (like where the cat really is), while query vectors handle John’s perspective (where he thinks the cat is). The key vectors ensure that the model compares John’s belief to reality, helping it detect the contradiction.

Finally, in the 'v' plot (value vectors), certain heads like head 3 in layers 15 and 25 are particularly important. Values are the actual information passed on after attention, so heads with impactful value vectors directly shape the model’s final output.

When we compare across the plots, a few heads consistently stand out, while others are more specialized—focusing on either queries, keys, or values. Head 5 at layer 20 impacts both the output and queries, while head 3 is more influential on keys and values. It’s fascinating to see how different heads specialize: some are more crucial for attending to the right tokens (through q and k), while others are focused on aggregating and transmitting information (through v and z). All these heads work together to guide the final output.

The temporal pattern is also worth noting—patching heads in the last 5-10 layers has much more impact than earlier layers. This aligns with the idea that early layers handle feature extraction, while later layers focus on resolving the final output.

<br>

## So What?
<sub>[Contents](#top)</sub>

<br>

The model seems to have developed a systematic, multi-step process for solving this task. It starts by identifying the key facts (like `cat on box`), integrates context, and then, in the final layers, resolves any ambiguity to arrive at the correct conclusion (that the cat should be in the basket). This structured approach suggests the model has learned a robust, step-by-step strategy.

We can see that earlier residual streams and later attention heads both play crucial roles. The model seems to handle basic token-level dependencies in the early layers, while deeper layers focus on more complex, context-driven reasoning. This pattern aligns with how transformers generally handle more intricate reasoning tasks—processing simple relationships early, and refining the understanding in later stages.

The sparse and localized computation in the activation patching plots show us that a few key tokens (like `box` and `the`) and a few attention heads across specific layers carry most of the important information. This sparsity signals efficiency and specialization in how the model processes the task, which was evident from the attention analysis.

Different heads specialize in distinct functions. **Some focus on attending to the right tokens** (through queries and keys), **while others are more important for aggregating and passing on information** (through values and outputs). This division of labor shows that the model breaks down the task into subtasks, with different heads handling different parts of the process.

What’s interesting is that a head’s role evolves over the layers. The output of a head at one layer isn’t just a simple transformation of what it did in the previous layer. There are complex interactions between heads and the residual stream, allowing the model to gradually shift its internal representation and get closer to solving the task as it moves through the layers.

The last few layers are particularly important for the final output—small tweaks here can shift the model’s prediction. This fits with the idea that earlier layers are mainly focused on feature extraction and building a representation, while the later layers are more about making the final decision. The model has learned how to transform its input into a form where making the final classification becomes straightforward.

Another interesting point is that patching just a few key components—either specific tokens or heads—with activations from a clean run is enough to steer the model back to the correct answer. This suggests the model’s understanding isn’t brittle. Rather, it can be "nudged" in the right direction by fixing a few critical pieces.

The model breaks the problem down into specialized subtasks, processes information in a sparse and localized way, and gradually transforms its representation over multiple layers to reach the right conclusion.

From the current observations we can begin to theorize how the model is performing ToM.

```markdown
### Tracking the Belief Holder ("John")

Initial State:
    - Layer 0, Head 3:
      - Q/K: Attends to "John."
      - V: Writes entity information.

Belief Construction:
    - Layer 14, Head 0:
      - Q/K: Integrates "John thinks."
      - V: Outputs "basket," linking John's perspective to the object's location.

### Moving Location Information

Initial Location Tracking:
    - Layer 0, Head 7:
      - Q/K: Focuses on prepositions.
      - V: High activation for "on," establishing spatial relationships.

Movement Tracking:
    - Layer 10, Head 1:
      - Q/K: Attends to location changes.
      - V: Outputs "on" and "cat," tracking movement.

State Updates:
    - Layer 16, Head 3:
      - Q/K: Focuses on "cat" and locations.
      - V: Updates object-location bindings.

### Belief State Integration

    - Layer 22, Head 4:
      - Q/K: Selects crucial context for John's belief.
      - V: Writes strong "basket" activations, solidifying the incorrect belief.

Copy Suppression:
    - Layer 23, Head 5:
      - Q/K: Applies negative modulation.
      - V: Inhibits outdated information.
```

Across each set of heads, the model continually refers back to foundational representations encoded in previous layers to update and refine the model's understanding across later layers, and it integrates information from different points in the narrative, from any position in the sequence, relying on earlier representations to interpret and refine later events correctly. For example, the initial identification of "John" as the belief holder is established early but remains relevant throughout the processing.

Across each set of heads, the model relies on earlier representations as foundational anchors that it keeps referring back to, updating and refining them as it moves through later layers. Heads in each layer pull from earlier encoded information to track the narrative and piece together context from different positions in the sequence. For instance, when the model identifies "John" as the belief holder early on, it continues to integrate that information across layers to correctly interpret events in later parts of the narrative. The same goes for any linguistic element.

<br>

### Dictionary learning, sparse autoencoders and superposition
<sub>[Contents](#top)</sub>

<br>

The linear representation hypothesis tells us that activations are **sparse**, **linear** combinations of **meaningful feature vectors**.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697598-f4d78700-4204-11eb-837c-40fb733f52df.gif" width="550px">
</p>

<br/>

Dictionary learning is closely related to the linear representation hypothesis, and allows complex data to be expressed as a linear combination of simpler elements. It can be used to break down data into simpler parts, which we call basis vectors. The goal is to find a small set of basis vectors that can efficiently describe the data, making it easier to analyze, compress, or reconstruct. These basis vectors form a "dictionary" of basic components that can be combined in different ways to recreate or represent the original data.

There is some dictionary (data structure for storing a group of things) of concepts that the model knows about—what it's learned during training—and each one has a direction associated with it. On a given input some of these concepts are relevant, they get some score and its activations are roughly linear combinations of those directions weighted by how important they are eg. king is the male direction + the royalty direction. Sparsity comes into play because most concepts are not relevant to most inputs, eg. royalty is irrelevant to bananas, so most of the feature scores will be 0.

Sparse autoencoders (SAEs) are neural networks that learn both the dictionary and the sparse vector of coefficients. The key idea is to train a wide autoencoder to reconstruct the input activations so that the hidden state learns the coefficients of the meaningful combinations of neurons and the decoder matrix—the dictionary—learns the meaningful feature vectors and each latent variable in the autoencoder is a different learned concept.

The hope is that if there is an interpretable sparse decomposition—the output of the mechanism the autoencoder is learning from—it is now human interpretable.

This technique allows us to find abstract features that the model uses to represent concepts that the model uses to make predictions. These features are causually meaningful, and we can steer the model's output (behavior). So SAEs find real structure in the model that shows us how it is performing a task.

Even simpler, we can think of them as microscopes that combat the curse of dimensionality and lets us see inside language models to better understand how they work.

SAEs are based on the hypothesis that models have a big list of concepts they "know" about, with associated directions. On each input, only a few concepts matter and model internals are linear combinations of those directions. SAEs help find these directions (mention directions in residual stream that are read/written by attention and MLPs). 

There are many directions to find because of **1)** polysemanticity, where many neurons fire for multiple, often times unrelated features.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/16ce1f4b-32dd-486d-920b-ae2394cea058" width="950px">
</p>

<br/>

And **2)** superposition, neural networks represent more concepts (features) than they have neurons and uses linear combinations of neurons to represent these concepts. 

Basically neurons represent multiple different things and features are spread across multiple different neurons. Because of superposition, we have a limited number of neurons for all our features, so there are lots of features and not so many neurons in any given activation space. But the irony is that the features are actually sparse, so only a few of them are active at any given time. This allows us to take advantage of SAEs. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/e7869efd-bcaa-4d21-b49f-c0e1db1de148" width="480"/>
</p>

<br>

So we can take the activation vectors from attention, an MLP or the residual stream, expand them in a wider space using the SAE where each dimension is a new feature and the wider space will be sparse, which allows us to reconstruct the original activation vector from the wider sparse space, then we get complex features that the attention, MLP and residual stream have learned from the input. From this we can extract rich structures and representations that the model has learned and how it thinks about different features as its processing the input.  

The SAE suite I used for this analysis is Google Deepmind's <a href="https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/" title="Google Deepmind" rel="nofollow">Gemma Scope</a>, and the output was visualized using <a href="https://docs.neuronpedia.org/" title="Neuronpedia" rel="nofollow">Neuronpedia</a>. Gemma Scope is a collection of hundreds of SAEs on every layer and sublayer of Gemma-2-2B and 9B. Using the trained SAE on the ToM passage, we can take features from layer 22 of Gemma-2-2B out of superposition, and see which features in the model are activated.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2c554f22-7de0-4b2b-9f5e-2a30faef77b3" width="480"/>
<img src="https://github.com/user-attachments/assets/323a6cb4-e431-4e3b-96f5-cb7073839dbd" width="480"/>
</p>

<br>

Looking at the residual stream features activated for the ToM passage, it seems like the model has specific features dedicated to representing different aspects of the narrative. For example, on a more granular level, feature 61 focuses on *references to positions and locations in a narrative*, feature 2704 captures *phrases or contexts involving going to a place or location*, and feature 3 seems tied to *objects or items typically associated with or placed on surfaces*. Each of these has high explanation scores<sub>[<a href="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain" title="Bills" rel="nofollow">17</a>]</sub>, showing that the model is isolating different narrative elements through distinct features.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/285680ab-c15e-46f6-9c3d-f963430fe969" width="480"/>
<img src="https://github.com/user-attachments/assets/73540c29-3935-4b85-aeff-7a2b65a738f7" width="480"/>
</p>

<br>

These features suggest that the model is building an internal representation of the physical setup described in the passage, tracking where objects and characters are placed. It’s also clear that several features are responsible for keeping track of John and Mark's movements and actions. For example, feature 11013 captures *mentions of specific individuals and their actions or states in personal narratives*, while feature 9665 focuses on *phrases that emphasize ongoing actions or conditions*. This shows how the model segments and organizes different narrative elements through distinct features.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/5382e695-ca33-4ede-9440-461bbc902bce" width="480"/>
<img src="https://github.com/user-attachments/assets/2662e86c-7bd7-4abb-b9bc-b59033d72044" width="480"/>
</p>

<br>

The model also has features representing changes in the scene. Feature 4308 is about *phrases related to the concept of taking action or steps*, feature 6169 focuses on *words related to leaving or departure*.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/545661b6-e6b5-46c0-b20a-5101d50c93a9" width="480"/>
<img src="https://github.com/user-attachments/assets/1864263d-e072-4c1b-8e87-c3104e70334b" width="480"/>
</p>

<br>

The model’s ability to track changes to the scene is especially clear in how it handles temporal sequencing, keeping a detailed record of the order of events. For example, feature 21706 captures *statements involving returning or coming back from a situation or event*, while feature 10097 tracks *the verb 'look' as part of phrases that encourage or denote attention*. This shows the model’s mechanism for understanding not just static states, but the flow of actions over time.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/b4bbc79f-f7d2-45de-a4f0-ca0582e204b7" width="480"/>
<img src="https://github.com/user-attachments/assets/ad4edeea-8b8f-4898-8712-12565522aece" width="280"/>
</p>

<br>

It seems like **the model also has features dedicated to representing "uncertainty" or "lacking knowledge"**. For instance, feature 9414 focuses on *phrases that begin with 'what' used in rhetorical or exclamatory contexts*, which could signal John’s lack of knowledge about what happened while he was away.

Since these representations were all recovered from the residual stream, we can see how it acts as a persistent information highway throughout the model’s layers, likely being further refined by the MLPs to capture more specialized information.

What’s interesting is that key information about the scene, characters, and their actions remains accessible across layers and can be picked up by either MLPs or attention heads. The residual stream typically carries a mix of information relevant to various aspects of language processing, and the presence of ToM-related features suggests the model is learning linguistic patterns tied to cognitive, spatial, temporal, and causal processes—core components of ToM tasks.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/dd981a3c-34df-44f4-8ac1-9f5089fd7d6e" width="480"/>
<img src="https://github.com/user-attachments/assets/7d762b3c-00ad-4ce6-b18b-e66e38631068" width="480"/>
</p>

<br>

The residual stream's nature allows for continuous updating of information, which is especially important in ToM scenarios, where belief states need to update as new information comes in. Features like 10427 (related to "capabilities and possibilities") and 11271 (focused on "inquiries or questions") being present in the residual stream suggest the model can dynamically adjust its representation of characters' belief states as it processes input.

The fact that ToM-related features show up in the residual stream points to Gemma-2-2B's approach to ToM as being highly integrated, distributed across layers, and dynamic. The model’s ability to update belief states on the fly is a key part of how it handles ToM tasks.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/35024155-99d1-4595-9ed1-2ab162c7580f" width="480"/>
<img src="https://github.com/user-attachments/assets/428ae4e5-c003-4404-a124-260cc593988e" width="480"/>
</p>

<br>

Looking at the MLP features, feature 11284 is tied to *verbs related to actions and states in a narrative context*. This likely helps the model process actions like John and Mark taking the cat, putting it on objects, and leaving the room. Feature 5852, on the other hand, is focused on *verbs and phrases related to physical observation or visual engagement*, which probably plays a key role in handling John’s final action of looking around the room. These MLP features seem to help the model handle specific actions and observations, grounding narrative events in a way that's useful for tasks like ToM.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/0b9644d0-f7e7-4bc1-b483-71b45f8eecf8" width="480"/>
<img src="https://github.com/user-attachments/assets/8b9c0fbf-9350-4b08-b2e0-f029e157bef7" width="480"/>
<img src="https://github.com/user-attachments/assets/c3ed915f-da78-4777-8c63-cdc206218901" width="480"/>
</p>

<br>

Feature 13597 seems to play a role in representing the individual experiences of John and Mark, while feature 7929 likely processes the characters' movements in and out of the room, potentially with a focus on tracking these transitions. Feature 12442 appears to be responsible for maintaining the spatial representation of the room and the objects within it. These features all contribute to how the model tracks and processes the physical and experiential aspects of the scene, which is key for understanding the dynamics of the narrative.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/7479b1e5-edc3-47f8-a6e6-151a8cf0469f" width="480"/>
<img src="https://github.com/user-attachments/assets/73123c93-98f3-43c5-845d-25b3f9c7b6b8" width="480"/>
<img src="https://github.com/user-attachments/assets/4e48f9d9-1a81-472a-a21b-b058be9335c3" width="480"/>
</p>

<br>

Several features seem to be directly tied to representing belief states and knowledge. Feature 13597 is likely crucial for capturing John's lack of knowledge about what happened in the room while he was away. Feature 5107 probably signals the model’s awareness of John’s ignorance, potentially reflecting uncertainty and doubt. Feature 12703 could be involved in modeling John’s thought process when he returns to the room, helping the model represent how John updates his beliefs. These features seem key for understanding how the model processes ToM scenarios, especially when tracking characters’ evolving mental states.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/69849c3b-1005-487f-8e7c-679d0d8a0ee8" width="480"/>
<img src="https://github.com/user-attachments/assets/ee4d1867-9512-4804-9f92-076db60ed459" width="480"/>
</p>

<br>

A recurring theme in the model’s processing of the passage is its focus on temporal sequencing and how events relate to one another. Feature 10766 seems to track states or conditions related to the timing of events or actions, while feature 3402 appears to deal with abstract temporal concepts. Feature 4320 likely identifies key moments in the narrative that trigger shifts in the scene or changes in the characters’ beliefs. These features suggest the model has mechanisms for keeping track of when things happen and how they influence the broader context—crucial for understanding the evolving dynamics.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/d4171c64-b7af-4ff9-83fa-36f8a4c0f03f" width="480"/>
<img src="https://github.com/user-attachments/assets/76a45f20-be69-4f2f-a1e7-d1e2a5f70eee" width="480"/>
<img src="https://github.com/user-attachments/assets/cf95b8e5-09ff-4753-8d9d-63575417fe1b" width="480"/>
<img src="https://github.com/user-attachments/assets/ac9db7c7-255a-4bf7-ae70-868f23c5a19d" width="280"/>
</p>

<br>

Another key aspect for the ToM task is spatial processing. Feature 12441 likely tracks the positions of the cat, box, and basket, while feature 346 seems to process how subjects and objects move around the room.

What’s pretty clear from this is that the MLP features show a high degree of specialization. ToM-related features are distributed across multiple distinct MLPs, suggesting the model doesn’t rely on a single "ToM module". Instead, it integrates various aspects of *reasoning* to achieve ToM understanding.

The features range from low-level tasks (like tracking object positions) to high-level abstractions (like representing uncertainty and beliefs), showing a hierarchical approach to processing the ToM scenario. The model also seems to maintain parallel representations of the actual state of the world and the characters' beliefs about it, which is key for solid processing of ToM tasks.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/02fec28d-0770-40d3-9546-c2375a4c9c7c" width="480"/>
    <br>
<small style="font-size: 8px;">Features like 13313 show that the model is capable of integrating contextual information to support its ToM abilities.</a></small>
</p>

<br>

Gemma-2-2B demonstrates a highly sophisticated and distributed approach to processing ToM scenarios. The model seems to have developed specialized concepts for handling various aspects of ToM processing, including belief representation, spatial awareness, temporal sequencing, and integrating contradictory information.

This allows the model to maintain multiple token representations simultaneously (like reality vs. belief) within each layer which enables it to maintain and update representations throughout the entire sequence globally, and handle complex ToM scenarios by integrating information across these specialized features. The fact that these features appear in the MLPs suggests that a lot of the heavy lifting for ToM is happening within the model’s feed-forward networks, which complement the information flowing through the residual stream.

What’s especially interesting is that these features represent cases where the model has learned specific behaviors it can then replicate. This highlights how powerful gradient descent is—it finds solutions and learns patterns that we wouldn’t even think to look for. That’s why SAEs are so useful here: they let us uncover the features gradient descent has already taught the model without needing to guess.

<br>

### ToM circuit <a id="tom-circuit"></a> 
<sub>[Contents](#top)</sub>

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/29b7c1e7-a97d-4600-a4d7-166beb7fab2d" width="650"/>
</p>

<br>

Iterative analysis of attention patterns and activation patching has revealed a lot about how ToM is represented and processed in a DOLM. The model performs a complex, but interpretable algorithm to perform this particular false-belief task, and it's based on a circuit involving 16 attention heads.

The circuit shows a clear hierarchical structure, breaking down into these components:

- **Initial State Heads** identify initial state of locations and subject positions.
    - e.g., cat in room, box in room, basket in room, John in room, Mark in room, room
      
- **Action State Heads** identify subject actions and their relationships to objects.
    - e.g., John puts cat on basket, Mark takes cat off basket, Mark puts cat on box
      
- **Scene Representation** integrates the initial states and actions, placing them in the context of the ongoing scene, and integrates location changes
    - e.g., John puts cat on basket then leaves room, Mark puts cat on box then leaves room, John returns to room
      
- **Belief State Emphasis Heads** maintain subject's mental state from subjects initial state.
    - e.g., John put cat on basket, John at school, Mark takes cat off basket, Mark put cat on box, John not in room, Mark at school, cat currently on box (according to Mark's belief), cat currently on basket (according to John's belief)
      
- **Copy Supression Heads** negatively effect true-beliefs and prevents copying the actual location of the object.
    - e.g., John+++, Mark+, cat on basket++++, cat on box--
 
<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/4169d5a0-b678-4971-b9df-cd421eeceadf" width="650"/>
<br>
<small style="font-size: 8px;">Theory of Mind Circuit.</a></small>
</p>

<br>



The data extracted from the attention mechanism looks something like this<sub>[<a href="https://github.com/christianThardy/christianThardy.github.io/blob/master/q-k-v-output.md" title="Hardy" rel="nofollow">22</a>]</sub>:



Its clear this layer focuses on basic function words and spatial relationships. 

This is generally true for the early layers, or initial state heads —they mostly handle simple things like basic linguistic elements (parts-of-speech: puncuation, determiners, conjugations. syntactic dependencies) in specialized later heads. Here, we often see stronger contributions from the key vectors, suggesting these layers are mostly about gathering broad contextual information and maintaining diffuse attention patterns.

As we move into the middle layers, things get more interesting. The actions state heads and scene representation heads start integrating information from the initial state heads. This is where object tracking, action understanding, and structural processing starts to form. The attention mechanism becomes more balanced between the query and key vectors, reflecting how the model is integrating information and refining its contextual understanding of the scene.

This integration feeds into the belief state heads, especially for entities like John and Mark, where the model begins to track complex subject-object interactions and manage belief states—continuing to maintain the broader context built up from earlier layers. It’s here that we see the emergence of complex reasoning, such as tracking belief states while keeping attention on earlier elements of the narrative.

At the final stages, the copy suppression heads play a key role. These heads show both positive and negative modulations between the QK mechanisms, working to manage information propagation. The value mechanism here kicks in to inhibit outdated or irrelevant information, ensuring only the relevant aspects—like a false belief about an object’s location—end up influencing the final prediction.

For instance, we can break down how the model builds a subject's false belief about an object’s location by: 1) Establishing John as the belief holder. 2) Tracking the cat's movement. 3) Updating object locations. 4) Integrating these elements into John's belief state. 5) Suppressing outdated or irrelevant information.





**MOVE TO CIRCUITS SECTION AND REFERENCE THIS LOGIT DIFFERENCE FROM EACH HEAD PLOT**
These heads correspond to some of the name mover heads (renamed location mover heads for this analysis) and negative name mover heads (renamed negative location mover heads for this analysis) discussed in the paper. There are also other heads that matter positively or negatively but to a lesser degree—these include additional location movers and backup location movers. More on this later.
**MOVE TO CIRCUITS SECTION AND REFERENCE THIS LOGIT DIFFERENCE FROM EACH HEAD PLOT**





 ^ Should be broken down even further, should be able to say which mech reads/attends then writes. It feels like I'm lacking specificity on how the QKV are interacting in the circuit during prediction.

Use the example here in the post, and link to the rest of the results in a different .md file
<br>

**Provide high and low level explanation of attention heads and their patterns that make up each node in the circuit**



In the last layer the model wants to focus on facts, but the facts are supressed



In layer 22 head 2 suggests global context consideration in prediction as the earlier mentioned basket tokens are heavily attended to by the key vectors (I think)

The model is taking John and Marks actions into consideration




**Do not forget to show the linguistic parallels**

The circuit as a whole is made up of various attention heads (induction and copy supression heads) that perform algorithms to move information from the context of the sentence 

<br>

### Copy supressions role in the ToM circuit
<sub>[Contents](#top)</sub>

<br>

Copy supression[<a href="https://arxiv.org/pdf/2310.04625" title="McDougall" rel="nofollow">19</a>] in the ToM circuit is a head in the model that responds to the predictions that are being made by heads in earlier layers and calibrating the final prediction. It's useful for later heads to do this because they get to see everything that comes before them. They get to see all of the context made by the earlier heads in the model and then adjust the level of confidence (positive/negative) of the next predicted token in the sequence wrt the logits before the final token is predicted.

More technically, copy surpression is an algorithm applied to the unembedding space of the model. An induction head (belief state emphasis) sees that `John put the cat on the basket`, the current token is `the` and it starts to output `basket`. This is written to the residual stream and will be mapped to the logits, but then copy supression performs post-processing on this logit space by supressing any output it has seen before that is not relevant to the induction heads context. So we can see heads that do task specific things and then heads that are responding to the previous predictions which is a more general and less specific sub task.

The amount of copy supression is mediated by the amount of attention paid to the copied token. Which makes sense, DOLMs iteratively refine their predictions by learning iteratively, being trained iteratively, then they represent information iteratively though each of its layers as information approaches the final layers.  

There's a lot more we do not know about these heads and they probably have more complex circuitry that describes when it is good to copy surpress information and when it is bad. 

(Replicate overconfidence metric analysis to test copy supression heads)

(Replicate qk, ov matrices of CS head to test its ability to produce the negative of box)

**MOVE TO CIRCUITS SECTION AND REFERENCE THIS LOGIT DIFFERENCE FROM EACH HEAD PLOT**
These heads correspond to some of the name mover heads (renamed location mover heads for this analysis) and negative name mover heads (renamed negative location mover heads for this analysis) discussed in the paper. There are also other heads that matter positively or negatively but to a lesser degree—these include additional location movers and backup location movers. More on this later.
**MOVE TO CIRCUITS SECTION AND REFERENCE THIS LOGIT DIFFERENCE FROM EACH HEAD PLOT**

<br>

### Ablation studies

<br>

# Conclusion <a id="conclusion"></a>
<sub>[Contents](#top)</sub>

<br>

The results should be taken with a grain of salt, as the model was only evaluated on one ToM passage. In a future update, goal is to run a proper ablation study on multiple passages to validate or invalidate the proposed circuit.

Grab the most compelling insights from each section and reiterate on them here

The activation patching plots show that the output prediction is strongly influenced by the attention heads that focus on John's last known action and the initial state of the room. Those attention heads are..........

The circuit is not very clean, its a cyclical/recursive task

<br>

<br>

# References:
<sub>[Contents](#top)</sub>

<br>

Mahowald, *Dissociating Language And Thought In Large Language Models.* University of Texas at Austin, Georgia Institute of Technology, UCLA, MIT. 2024.[<a href="https://arxiv.org/pdf/2301.06627" title="Mahowald" rel="nofollow">1</a>]

Jamali, *Semantic encoding during language comprehension at single-cell resolution.* Nature. 2023.[<a href="https://www.nature.com/articles/s41586-024-07643-2" title="Jamali" rel="nofollow">2</a>]

Kosinski, *Evaluating Large Language Models in Theory of Mind Tasks.* Stanford University. 2023.[<a href="https://arxiv.org/pdf/2302.02083" title="Kosinski" rel="nofollow">3</a>]

Ullman, *Large Language Models Fail on Trivial Alterations to Theory-of-Mind Tasks.* Harvard. 2023.[<a href="https://arxiv.org/pdf/2302.02083" title="Ullman" rel="nofollow">4</a>]

Oguntola, *Deep Interpretable Models of Theory of Mind.*  Carnegie Mellon University. 2021.[<a href="https://www.nature.com/articles/s41586-024-07643-2" title="Oguntola" rel="nofollow">5</a>]

Le, *Revisiting the Evaluation of Theory of Mind through Question Answering.* Facebook AI Research. 2019.[<a href="https://aclanthology.org/D19-1598.pdf" title="Le" rel="nofollow">6</a>]

Ma, *Towards A Holistic Landscape of Situated Theory of Mind in Large Language Models.* University of Michigan. 2023.[<a href="https://arxiv.org/pdf/2310.19619" title="Ma" rel="nofollow">7</a>]

Jamali, *Unveiling theory of mind in large language models: A parallel tosingle neurons in the human brain.* Harvard. 2023.[<a href="https://arxiv.org/pdf/2309.01660" title="Jamali" rel="nofollow">8</a>]

Nguyen, *Language Models are Bounded Pragmatic Speakers: Understanding RLHF from a Bayesian Cognitive Modeling Perspective.* 2024.[<a href="https://arxiv.org/pdf/2305.17760" title="Nguyen" rel="nofollow">9</a>]

Wang, *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* Redwood Research, UC Berkley. 2022.[<a href="https://arxiv.org/pdf/2211.00593" title="Wang" rel="nofollow">10</a>] 

Park, *The Linear Representation Hypothesis and the Geometry of Large Language Models.* 2024.[<a href="https://arxiv.org/pdf/2211.00593" title="Park" rel="nofollow">11</a>] 

Mikolov, *Linguistic Regularities in Continuous Space Word Representations.* Microsoft Research. 2013.[<a href="https://aclanthology.org/N13-1090.pdf" title="Mikolov" rel="nofollow">12</a>]

Htut, *Do Attention Heads in BERT Track Syntactic Dependencies?* NYU. 2019.[<a href="https://arxiv.org/pdf/1911.12246" title="Htut" rel="nofollow">13</a>]

Yun, *Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors.* Facebook AI Research, UC Berkley, NYU. 2023.[<a href="https://arxiv.org/pdf/2103.15949" title="Yun" rel="nofollow">14</a>]

Riggs, *Really Strong Features Found in Residual Stream.* 2023.[<a href="https://www.lesswrong.com/posts/Q76CpqHeEMykKpFdB/really-strong-features-found-in-residual-stream" title="Riggs" rel="nofollow">15</a>]

Elhage, *A Mathematical Framework for Transformer Circuits* Anthropic. 2021.[<a href="https://transformer-circuits.pub/2021/framework/index.html#residual-comms/" title="Elhage" rel="nofollow">16</a>]

Bricken, *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* Anthropic. 2023.[<a href="https://transformer-circuits.pub/2023/monosemantic-features/index.html" title="Bricken" rel="nofollow">17</a>]

Bills, *Language models can explain neurons in language models* OpenAI. 2023.[<a href="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain" title="Bills" rel="nofollow">18</a>]

Cunningham, *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* EleutherAI, MATS, Bristol AI Safety Centre, Apollo Research. 2023.[<a href="https://arxiv.org/pdf/2309.08600" title="Cunningham" rel="nofollow">19</a>]

Templeton, *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Anthropic. 2024.[<a href="https://transformer-circuits.pub/2024/scaling-monosemanticity/" title="Templeton" rel="nofollow">20</a>]

McDougall, *Copy Suppression: Comphrehensively Understanding an Attention Head.* Independent, University of Texas, Google Deepmind. 2024.[<a href="https://arxiv.org/pdf/2310.04625" title="McDougall" rel="nofollow">21</a>]
