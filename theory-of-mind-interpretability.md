# Deconstructing theory of mind in large language models

<br>

#### Table of Contents <a id="top"></a>
- [Introduction](#introduction)
- [The relationship between theory of mind and language](#the-relationship-between-theory-of-mind-and-language)
    - [So What?](#so-what)
- [Theory of Mind Circuit Discovery](#theory-of-mind-circuit-discovery)
    - [Principal Component Analysis](#principal-component-analysis)
    - [Identify Relevant Layers and Activations](#identify-relevant-layers-and-activations)
    - [Residual Stream and Multi-Head Attention](#residual-stream-and-multi-head-attention)
    - [Iterative Attention Head Analysis and Causal Tracing](#iterative-attention-head-analysis-and-causal-tracing)
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

<a href="https://arxiv.org/pdf/2407.02646" title="arxiv" rel="nofollow">Mechanistic interpretability</a> gives us a way to reverse engineer the internal workings of neural networks, turning the representations they learn into understandable algorithms. This helps us trace which parts of the model matter for a given task and decompose paths within the model into interpretable components that we can reason about, piece by piece. It's like having an x-ray or microscope to see inside the model. It reveals not just how model’s make decisions but also why certain behaviors stick around, even when we're trying to fine-tune or align them.

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

To explore how ToM could be represented algorithmically, we will dig into the linguistic principles of **semantics**, and **pragmatics** in the context of this false belief passage: 

*'In the room there are John, Mark, a cat, a box, and a basket. John takes the cat and puts it on the basket. He leaves the room and goes to school. While John is away, Mark takes the cat off the basket and puts it on the box. Mark leaves the room and goes to work. John comes back from school and enters the room. John looks around the room. He doesn’t know what happened in the room when he was away. John thinks the cat is on the...'*

<br>

### Semantics

Semantics is all about representing meaning in language. It focuses on how words, phrases, and sentences convey meaning, and how humans interpret that meaning. It’s not just about the surface-level meaning of words, but also how those meanings combine and interact in context. Semantics covers a lot of ground, including things like compositional semantics, semantic similarity, word embeddings, distributional semantics, and distributed semantics.

For example, to linguistically understand the semantics of the ToM passage, we need to identify the entities, actions, relationships, and any implied meanings to correctly predict the final token `basket`. To do this, we need to break down the sentence into all its entities and actions and map out how they interact. This is crucial for making sense of what's happening, especially when dealing with more abstract reasoning like ToM.

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
<sub>[↑](#top)</sub>

These principles and operations can *help* interpret how humans perform ToM linguistically, but how do these concepts transfer to large language models in relation to ToM? 

By being trained for next word prediction, LLMs end up learning a lot about the structure of language, including linguistic features that were, until recently, thought to be out of reach for statistical models.

For example, a common way to test linguistic abstraction in LLMs is through probing. This involves training a classifier on internal model representations to predict abstract categories, like part-of-speech or dependency roles. The goal is to see whether these abstract categories can be recovered from the model’s internal states. Using this method, researchers have claimed that LLMs essentially "rediscover the classical NLP pipeline," learning linguistic features like part-of-speech tags, parse trees, and semantic roles across different layers.

ToM prediction heavily relies on context to make sense of the mental states and intentions behind the words and actions of others, and final word prediction is based on implied meanings (implicature) and inferred intentions (presupposition), which are central to pragmatics. Given the literature, even if the phenomena just statistical, **some** form of semantic and pragmatic inference in LLMs has been learned, regardless of how uneven or weak the performance.

<br>

# Theory of mind circuit discovery
<sub>[↑](#top)</sub>

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
<sub>[↑](#top)</sub>

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
<sub>[↑](#top)</sub>

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

Based on what I've seen with PCA, I think its possible that the ToM task may be aligned with the linear representation hypothesis<sub>[<a href="https://arxiv.org/pdf/2311.03658" title="Park" rel="nofollow">11</a>]</sub><sub>[<a href="https://aclanthology.org/N13-1090.pdf" title="Mikolov" rel="nofollow">12</a>]</sub> –the idea that models pick up properties of the input and represent them as directions in activation space. When we dig into layer 22's PCA, a few interesting things stand out.

The PCA breaks down into three clusters of concepts:

- Actor tokens (`John`, `Mark`, `cat`)
- Mental state tokens (`thinks`, `knows`)
- Location tokens (`basket`, `box`, `room`)

In the residual stream pre, there is clustering of scene elements and characters, and the separation between different semantic groups looks linear.


In the residual stream pre, scene elements and characters begin clustering, but in the residual stream post (shared space where all layers interact) the separation is even clearer,  aligning these clusters more tightly around token concepts:

- `John` and `thinks`
- `basket` and initial state
- `box` and current state


This clearer clustering reinforces the updated relationship between the MLP layers and residual stream, where we now see distinctions between knowledge states (e.g., what John knows vs. doesn’t know) mapped linearly. This makes sense because if tokens didn’t cluster within residual space, then linear transformations across layers would be less informative.

The clustering remains clear as the attention and MLP layer outputs are added back to the residual stream with updated relationships. The separation of "knowledge states" (what John knows vs what he does not) appears linear. The spatial relationships also look linear. If information isn't clustered in residual stream space, linear operations between layers wouldn't be meaningful.

When we compare this with the linear plot above, its clear that the model is keeping two separate but parallel "tracks":

- Reality track (blue): represents actual events
- Belief track (red): represents John's belief state

The key thing here is that after Mark moves the cat, the two tracks split, but the belief track stays locked into John’s original understanding. This suggests that the model is able to maintain two simultaneous yet distinct states—reality and belief—keeping them separate but interrelated to maintain parallel states. Even as the sequence progresses—Mark and John’s actions, them leaving, returning—the belief state remains consistent.

What’s also cool is that the PCA reveals these token clusters at consistently distinct distances, showing the same grouping across transformations. There’s almost a hypothetical “boundary” within the MLP and residual post layers, cleanly dividing what the model has learned about `John`, `Mark`, and their connection to the `basket`.

<br>

### Residual stream and multi-head attention
<sub>[↑](#top)</sub>

Attention heads are valuable to study because we can directly analyze their attention patterns—basically, we can see which positions they pull information from and where they move it to. This is especially helpful in our case since we're focused on the logits, meaning we can just look at the attention patterns from the final token to understand their direct impact.

One common mistake when interpreting attention patterns is to assume that the heads are paying attention to the token itself—maybe trying to account for its meaning or context. But really, all we know for sure is that attention heads move information from the residual stream at the position of that token. Especially in later layers, the residual stream might hold information that has nothing to do with the literal token at that position! For example, the period at the end of a sentence might store summary information for the entire sentence up to that point. So when a head attends to it, it’s likely moving that summary information, not caring if it ends with punctuation. This makes it hard to asses what the attention heads are doing when tokens are being attended to. 

But at the same time, I think when an attention head is attending to a token, it is accessing abstract information stored at that position.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/c64273c7-5d0b-4efc-bbd1-b0ed05842aa5" width="280"/>
</p>

<br>

In transformer architectures, each token position has a residual stream—a vector that carries forward information as the model processes each layer. We can think of the residual stream as the place where everything communicated from earlier layers are communicated to later layers. It aggregates outputs from previous attention heads and MLPs—everything the model has *thought* so far.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/d1590634-0cb0-42f5-b177-a17ee0203af1" width="280"/>
</p>

<br>

Both attention heads and MLPs read from this stream, apply their edits, and then write the modified info back into the residual stream using linear operations (just simple addition). This linearity is key—it allows the input to any layer be decomposed as the sum of contributions from various mechanisms across different layers.

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

The model is using the residual stream to achieve compositionality between different pieces of information. For example, there could be some attention head in layer 2 that composes with some head in layer 22. Technically this looks like some head in the 1st layer will output some vector to the residual stream, the head in the 2nd layer will take as an input the entire residual stream and mostly focus on the output of the 1st layer and run some computation on it. For any pair of composing pieces in the model, they are completely free to choose their own interpretation of the input, so there's no reason that the encoding of the information between head 0 in layer 0 and head 5 in layer 3 will be the same as the encoding between head 2 in layer 0 and head 3 in layer 1. While extremely useful, this means we can expect the residual stream to be very difficult to interpret.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/415c58d0-6975-4483-808c-31cccf887cd9" width="7500"/>  
</p>

<br>

So, what’s happening here is the model builds up hierarchical representations of language—phrases within sentences, sentences within paragraphs—and tracks sequences of events, which is particularly important for tasks like ToM, where understanding the events, the order of events, character actions and possibly even directional or spatial information is key. 

In this framework, attention heads work like routers, directing specific pieces of information to the right places to solve the task. They aren’t just focusing on literal tokens but transferring abstract concepts like *"the last place John saw the cat"*, which aren't tied to any single token but are encoded in the residual stream.

This kind of hierarchical, nested structure in the residual stream is key to solving the ToM task. It requires the model to track what each character knows or believes over time, which means keeping updated representations of these abstract knowledge states in the residual stream.

In any case, it’s easy to get tricked if you think an attention head is just focusing on a literal token. We should be looking at this information alongside the information stored in the residual streams at that position—which often contains abstract concepts or higher-level representations.

While keeping all of that in mind, when looking at the plots, it’s a good time to start thinking about the algorithms the model might be using. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/9a91c747-f3f6-47ad-8ecd-5124dbcbc79f"/>
<img src="https://github.com/user-attachments/assets/0492e03e-66de-49f3-af70-45918d8efc93"/>
<img src="https://github.com/user-attachments/assets/64a36cf9-5bc7-4212-ba60-08f08eb4a12a"/>
<img src="https://github.com/user-attachments/assets/f680eed9-8fe9-4636-9bd2-736f4a10424c"/>
    <small style="font-size: 8px;">Attention patterns of the heads. We can see where each token attends by the maximum value of where its attending, tokens weighted by how much information is being copied, and how much every token effects every other token.</a></small>
</p>

<br>

We can start to connect the dots between earlier observations on semantics and pragmatics, and how they might show up in the model's attention patterns. We see that the model’s attention focuses on specific instances of the `basket`, especially when `John` is the only one interacting with it. This hints at the model potentially locking onto a key relation—between the subject `John`, the object `basket`, and the location—tied to those specific interaction moments.

This attention pattern suggests the model is encoding subject-object-location agreement and becoming more prominent in cases where the interaction is clear and exclusive to John. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2a42142b-e522-4469-91b4-e8470dba85da"/>
</p>

<br>

Here we see that the model is attending from the token `from` in the phrase `John comes back from school` to `school`, which appears earlier in the sentence. This demonstrates how the model links John's initial departure with his return, capturing continuity in the narrative. The model is utilizing previously seen tokens, such as `school`, to inform its current processing, and copying it to the current position. So the model's capability to reference earlier events aligns its understanding of John’s absence and return.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/a1ff26f7-85ff-42f5-8cbf-7028cd6ff6a5"/>
</p>

<br>

When coloring everything from the source token to the output token to see how much every token effects every other token, we get all the tokens that have their probabilities increased by the attention heads. We can see that `the` attended back to `box`, `basket`, `cat` which increased the probability that the next token should be `room`, suggesting noun phrases and more complex compositional patterns in the future.

We won’t dive into a full hypothesis about how the model works just yet—more on that later—but these are the kind of questions and iterative attention analysis that set the stage for figuring out the underlying circuit.

<br>

### Iterative attention head analysis and causal tracing <a id="iterative-attention-head-analysis-and-causal-tracing"></a> 
<sub>[↑](#top)</sub>

To trace which parts of the model's attention are key for this task, and break down those pathways, we need a deeper dive into the attention patterns. Specifically, we want to see how the model attends to tokens related to John, his initial actions, and his final actions.

One approach is tracking the activations of key tokens (John, basket, box, cat) across layers, showing how their representations evolve. Another approach is pinpointing which attention heads contribute most to predicting "basket."

By combining these methods and comparing the results, we can zero in on heads that attend to both the initial state and John’s final action.

Looking at the most basic units of computation in the attentions heads will give the most fine-grained account of what is happening when the model is processing information to be sent to the MLPs. So we need to explore the roles of the query (Q), key (K), and value (V) vectors across the hierarchy of layers.

The DOLMs attention mechanisms weigh the importance of different parts of the ToM passage. Each attention head computes three components:

- **Query (Q):** Determines which token positions to attend to.
- **Key (K):** Represents the tokens considered for attention at each position.
- **Value (V):** Contains the information to be propagated forward. Basically a weight determining to how relevant the key is to the query.

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
<small style="font-size: 8px;">Left to right: Establishing initial associations summarizing at final token position, preserving facts about the scene, preserving facts about the scene w/focus on initial context, encoding belief-related information in context & summarizing at final token position, preserving facts about the scene, tracking perspectives related to objects and actions, tracking perspective-based understanding and factual states, tracking belief states.</a></small>
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

We can sort of see evidence for copying heads (attend to a token and increase the probability of that token occuring again) in L0H7 and L10H1. Both showing rigid, position-based patterns, clean isolated spikes. L0H7 shows strong Q spikes at regular intervals with minimal KV interference, it might be doing token-level copying or positional tracking, but the sharp, forward, diagnoal increased magnitude of Q spikes screams systematic copying with position awareness to me. L10H1 shows copy-like behavior for specific syntactic structures with regular patterns around sentence boundaries and copying verb-related information forward.

Evidence for <a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" title="Olsson" rel="nofollow">induction heads</a> (look at present token in context, look back at similar things that have happened, predicts what will happen next) in layer 14 head 0 and layer 17 head 3. Both showing more flexible semantic-based patterns, and sharp, backwards K spikes and slight sharp forwards Q spikes. The former shows strong QK spikes at semantically similar tokens, attention to repeated patterns of actions/states, and the latter showing the tracking of recurring patterns in character actions, and next state predictions based on previous patterns.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/43905290-8648-435d-820a-9526d971fe0a" width="700"/>
<br>
</p>

<br>

Specifically, for the asymmetric patterns in layer 22 head 4, the highest Q attention (blue spike) is at the beginning of the sequence, around `basket` in the first mention of the basket, maybe suggesting the model is strongly querying the initial state of where the cat was placed (might be an artifact given its almost everywhere). The V attention (green) show strong contributions around `basket` early in the sequence, completely dominating the V attention of `box`, and several medium-height spikes around key events in the story (like when the cat is moved).

The pattern shows the model is attending strongly to both the initial state (`cat on basket`) and the intermediate state (`cat moved to box`). The high query attention to the initial `basket` placement suggests the model understands this is relevant to John's belief state, and even captures `John` in the initial state with high attention activations relative to `Mark`. The value contributions from both `basket` and `box` mentions show the model is tracking both possible locations of the cat; the real state (`cat on box`) and John's believed state (`cat on basket`), with the highest value contributions emphasizing tokens important to resolving the false belief and passing that information forward to other layers and heads. 

The strong attention to the initial state makes sense since that's what John last saw before leaving. The model also appears to be using this head to integrate information about object locations and character knowledge states. This head is likely key in some belief state emphasis context, and likely follows a collection of heads that build up to this attending to John's false belief. 

<br>

More formally, for each token position we have QKV vectors, 

Q<sub>i</sub> K<sub>i</sub> V<sub>i</sub>

<br>

And the attention score for the tokens position to another positions,

*score*(i,j) = *softmax*((Q<sub>i</sub> · K<sub>j</sub>) / √d<sub>k</sub>)

<br>

And output for position i is,

out<sub>i</sub> = Σ<sub>j</sub>(*score*(i,j) × V<sub>j</sub>

<br>

For the 4th head of layer 22 , the QKV vectors for the attention mechanism will look something like this,

Q<sub>basket</sub> ≈ 1.0 (tall blue spike)
K<sub>basket</sub> ≈ 0.3 (red line)
V<sub>basket</sub> ≈ 0.8 (tall green spike)

K<sub>box</sub> ≈ 0.2 (red line)
V<sub>box</sub> ≈ 0.4 (medium green spike)

*score*(basket,basket) = *softmax*((Q<sub>basket</sub> · K<sub>basket</sub>) / √d<sub>k</sub>)
≈ *softmax*((1.0 × 0.3) / √64)

*score*(basket,box) = *softmax*((Q<sub>basket</sub> · K<sub>box</sub>) / √d<sub>k</sub>)
≈ *softmax*((1.0 × 0.2) / √64)

out<sub>basket</sub> = *score*(basket,basket) × V<sub>basket</sub> + *score*(basket,box) × V<sub>box</sub>

out<sub>basket</sub> = *score*(basket,basket) × V<sub>basket</sub> + *score*(basket,box) × V<sub>box</sub>

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
- L23H5 and head 6 show particularly strong attention to `basket` over all instances of the token in the sequence, where `box` activations are relatively low.

**Activation patterns:**

- In the final layers (22-25), `basket` consistently has higher activation than `box`, despite `box` being the actual current location of the `cat`.

<br>

#### Causal Tracing: Activation patching

Activation patching is a super useful technique where internal activations in a neural network are replaced to target specific model behaviors and circuits. It allows us to choose which part to change so we can learn more about it.

The obvious limitation of the techniques we’ve used so far is that they only focus on the final parts of the circuit—the bits that directly affect the logits. That’s useful, but clearly not enough to fully understand the whole circuit. What we really want is to figure out how everything composes together to produce the final output, and ideally, we’d like to build an end-to-end circuit that explains the entire behavior.

This is where activation patching comes in. First introduced in the ROME paper (where they called it *causal tracing*), activation patching lets us dig deeper into the model’s internal computations. Here’s how it works:

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/aa21ea4f-67e4-4ab6-a373-cac81c8a3ee5" width="700"/>
<br>
<small style="font-size: 8px;">Patching into a transformer can be done in a bunch of different ways (e.g. values of the residual stream, the MLP, or attention heads' output). If you want to get really granular, you can patch at specific sequence positions (not shown). This flexibility lets us explore different components of the model and figure out exactly where certain behaviors are coming from.</a></small>
</p>

<br>

You run the model twice—once with a *clean* input that produces the correct answer, and once with a *corrupted* input that doesn’t. The trick is that during the corrupted run, you intervene by patching in an activation from the clean run at a specific point in the network. Basically, you replace the corrupted activation at a certain layer and position with the corresponding clean activation and then let the model continue its computation. The key insight here is that you can measure how much this patch shifts the output toward the correct answer, we can then assess the importance of that particular activation.

By iterating over lots of different activations, you can map out which ones matter. If patching a certain activation makes a big difference in pushing the model toward the right answer, it tells us that activation is important for the task. In other words, activation patching functions as a noising algorithm, contrasting with the denoising approaches we've previously focused on. In this approach, we run the model on a clean input but introduce "noise" by patching in activations from the corrupted run. The flip side is denoising, where we start with a corrupted input and patch in activations from the clean run, effectively removing noise.

With noising, just because performance drops when you ablate a component doesn’t automatically mean it was necessary for the task. For example, if you ablate layer 0 in Gemma-2-2B, performance gets much worse across a bunch of tasks, but that doesn’t mean layer 0 is crucial for the ToM task. In fact, it seems to function more like an extended embedding layer—useful for processing tokens but isn’t doing anything specific to ToM. We’ll dig deeper into this later, but the key point is that noising can lead to some ambiguous results, while denoising tends to give clearer answers.

The ability to localize computations like this is a huge win for mechanistic interpretability. If the model’s computations are spread out all over the place, it’s going to be much harder to form a clean, understandable story of what’s going on. But if we can pinpoint exactly which parts of the model matter, we can zoom in, figure out what they’re representing, how they’re connected, and ultimately have another super useful tool that we can use to reverse-engineer the circuit responsible for the observed behavior.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/e9680ec6-8c8e-4afe-90a6-0b20c59c53d4" width="500">
</p>

<br/>

- L22H4 shows a large positive logit difference, indicating that this head is crucial for the final prediction of `basket`.
- There are lots of negative contributions throughout the model, but L14H3, L16H2, and L23H5 are very negative and possibly components to a supression circuit (inhibition, negative mover) that helps the model focus on maintaining John's believed state.

An important thing to note is that these functions are not neatly isolated but distributed and overlapping across multiple positive and negative attention heads. For instance, several heads probably work together to represent the "mental state," and many of these heads also contribute to other tasks. The suppression activity, for example, doesn’t come from a single head—it emerges from the interactions between multiple heads throughout the network.

**REINFORCE THIS SECTION AFTER QKV ANALYSIS IS COMPLETE**
Specifically L8H6, L16H2, L18H7, and L23H5. All empirically show evidence of negative behavior on the final prediction as seen in the activation patching section. Each head has strong Q attention and low V attention to the `box` token, the most and strongest activations are happening in the middle of the sequence when Mark is moving the cat to the box.
**REINFORCE THIS SECTION AFTER QKV ANALYSIS IS COMPLETE**

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/a424bb3e-90f7-4992-ab3f-3fd26ba45ebe" width="900">
  <img src = "https://github.com/user-attachments/assets/1670b458-7601-4449-b901-3df3e706dfac" width="900">
</p>

<br/>

Diving deeper into the activation patching results, focusing on the residual stream/midstream, blue regions indicate where patching helped the model get closer to the correct prediction `basket`, while red regions show where patching hurt (pushing it towards `box`). The clean run is the uncorrupted input—where the model gets things right (`John thinks the cat is on the basket`). The corrupted run comes from swapping adjacent tokens, which messes up the sentence’s meaning and leads to wrong answers. The goal is to patch activations from the clean run into the corrupted one at various layers and sequence positions and see how much it improves the model’s logit difference (i.e., how much closer it gets to predicting the correct answer).

Patching the `box` token at layer 1 gives a massive boost, almost recovering full performance. But, as we move to later layers, the **most impactful patching** happens at the final `the` token before the blank where the model's prediction would go. **This shift hints at something important:** the model first focuses on where the `cat` was (`on the box`), and later on, it shifts to what word needs to be filled in (`basket` vs. `box`). There’s a super interesting pattern—starting from the `box` token in layer 0 and running up to the final `the` token in layer 25. This implies a distinct computational flow across the model’s layers. Early on, (layers 0-10) it’s all about the `box` token (likely where the model locks in the idea that the cat was on the box).

 Between layers 10-20, the patching impact spreads more evenly across tokens. This is probably where the model’s pulling everything together, building up a complete understanding of what’s going on and learning about the `box` vs `basket` contradiction. Then, by layers 20-25, the focus shifts hard onto the final `the` token—this is where the model's deciding which word (`basket` vs. `box`) to predict. While patching `box` is super helpful in early layers, it starts to hurt later on (negative blue regions). It seems like **the model needs to remember the original cat position** (`box`) early on but **then "forget" it** by the end to make the right call (`basket`). This shows how the model's thinking evolves layer by layer.

One cool takeaway is how localized the effect is—patching just a few tokens or layers can fix a lot of the model’s mistakes. It’s not spreading out the info evenly across the whole network. Instead, there’s a very directed flow of information from `box` to `the` over time.

**This fits with the bigger picture:** earlier layers are encoding the critical scene details (e.g., Mark moving the cat), while midstream activations are key for representing changes in location (whether the cat ends up on the basket or box).

The whole process aligns with previous attention analyses—early layers set up the scene, mid layers handle object movement and maintaining the scene, and late layers focus on reinforcing John’s false belief.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/b4d4f65b-6628-42a3-bb4b-de89915c8b82" width="950">
</p>

<br/>

The middle plot shows the effect of patching attention head outputs at each layer and head. Again, the color shows how much patching that specific head shifts the model's prediction from `box` to `basket`.

There are moderate signals in early layers, with the strongest signals being in the middle, suggesting the middle layers are important to the models' resolution of where the cat is located. Which contrasts to previous analysis of layer 22 where it seemed to be the most important. From this perspective it looks much more subtle, indicating that layer 22 might be doing more fine-tuning rather than making dramatic changes to the prediction. It's particularly striking because this pattern appears consistent across all three views (residual stream, attention output, and MLP output), where the signal from middle to later layers shifts more positive.

The biggest takeaway? The early layers are doing much of the heavy lifting in terms of building the representation and middle to late layers are doing much more modifying. This fits the pattern we’d expect—later layers are where the model locks in its final decision; small tweaks to attention outputs here can dramatically change the model’s output, whereas earlier layers are more about building up representations.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/cf67fbe1-e894-4118-9473-52c98f41d881" width="1000">
</p>

<br/>

These activation patching results shed light on what’s happening inside attention heads across layers. Again, each attention head does two key things: **1)** deciding where to move information (governed by the attention pattern, which the QK vectors control) and **2)** deciding what information to move (handled by the V vectors, influenced by the OV vectors). By isolating either the attention pattern or the value vectors through patching, we can tease apart which factor is more crucial.

Let’s start with the `z` plot (output vector). Patching outputs from certain heads noticeably  shifts the models' output from `box` to `basket`, particularly in the last 5-10 layers. The behavior is very distributed, but specifically, L16H7, L17H6, L22H2, L22H4 and L25H4 have the largest positive impact, along with L0H1, L3H1, L6H1, L8H1 (all previous layers have the same head, very interesting), L12H2, L14H3, L17H3, L16H2, L20H2, and L23H5 having the largest negative impact. Now, looking at the `q` plot (Q vectors), we see familar negative heads. This suggests that modifying the focus of these queries is pretty impactful for steering the model away from less accurate outputs. This signal pops up across early, mid, and late layers, possibly as the model navigates true belief alignment. The `k` plot (K vectors) is less clear, though L14H1 and L17H2 seem to matter. For the `v` plot (V vectors), certain heads like L22H1 and L22H2 are particularly important. Vs are the actual information passed on after attention, so heads with impactful V vectors directly shape the model’s final output.

When we compare across the plots, a few heads consistently stand out, while others are more specialized—focusing on either Qs, Ks, or Vs. L23H5 impacts both the output and Qs, while head 2 is more influential on Ks and Vs. It’s fascinating to see how different heads specialize: some heads prioritize token alignment (through Q and K), while others are focused on aggregating and relaying information (through V and Z).

Examining Q plot, layer 17 head 3, the patched activation is indicating a subtle negative activation. In the same layer and head's QKV plot from the iterative attention head analysis, `box` shows the highest activations with lower activations for `John` early in the sequence. Moderate activations for initial context and action words throughout the sequence. This suggests the model is actively balancing between different perspectives and factual grounding (the belief vs. reality contrast). 

By this point in the sequence, it seems the model has deduced that `box` doesn’t heavily attend to `John`, a relationship the patch highlights with that faint red activation. It’s also worth noting how this insight carries forward, flowing into layer 22, head 4, where the model continues to refine this nuanced “belief adjustment” with highest confidence as it learns.

Broadly speaking, models learn to interpret contextual relationships, which helps them understand cause and effect (John initially places the cat on the basket and then leaves the room, Mark then moves the cat to the box after John has left). Their Qs and Ks work together to determine which parts of the input each head attends to. 

Given the previous attention head analysis, it's plausible that Qs and Ks encode separate perspectives—where Qs represent John’s mental model of the cat’s location, while Ks capture the reality, and Vs could carry the false or true belief (where the cat really is), with Zs balancing the belief that is more weighted by the Q or K. Together, they nudge the model toward a coherent final output.

<br>

#### Causal Tracing: Path patching

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/6003f28f-a060-4de1-aa8e-ec86cddad86e" width="500">
</p>

<br/>

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/3919411e-7c83-42de-8ae7-f73fa567dd7f" width="500">
</p>

<br/>

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/a3d4650a-70c8-4af6-80aa-6497e49df6c2" width="700">
</p>

<br/>



<br>

## So What?
<sub>[↑](#top)</sub>

The model seems to have developed a systematic, multi-step process for solving this task. It starts by handling basic syntactic dependencies in the early layers, context-driven processing in middle layers by identifying the key facts (like `cat on box`), integrates that context into the final layers, resolves any ambiguity to arrive at the correct conclusion (`cat on basket`) using semantic attention patterns.

Different heads specialize in distinct functions. Particularly in layer 22 head 4, the head focuses on attending to important tokens that **compose and maintain perspectives** (queries, John's belief), **represent actual changes** in the environment **regardless of character perspective** (key vectors → state of the world), and **prioritizes groundtruth details** (values vectors → relevant details, where the output vector reflects whichever perspective the attention mechanism emphasizes), showing the models' capability to separate John's belief from reality. This division of labor shows that the model breaks down the task into subtasks, with different heads handling different parts of the process.

What’s interesting is that the role the head’s take over evolves across layers. The output of a head at one layer isn’t just a simple transformation of what it did in the previous layer. There are complex interactions between heads and the residual stream, allowing the model to gradually shift its internal representation and get closer to solving the task as it moves through the layers.

The last few layers are particularly important for the final output—small tweaks here can shift the model’s prediction. This fits with the idea that earlier layers are mainly focused on feature extraction and building a representation, while the later layers are more about making the final decision. The model has learned how to transform its input into a form where making the final classification becomes straightforward.

Another interesting point is that patching just a few key components—either specific tokens or heads—with activations from a clean run is enough to steer the model back to the correct answer. This suggests the model’s understanding isn’t brittle. Rather, it can be "nudged" in the right direction by fixing a few critical pieces, because it breaks the problem down into specialized subtasks, processes information in a sparse and localized way, and gradually transforms its representation over multiple layers to reach the right conclusion.

So across each set of heads, the model keeps circling back to foundational representations it encoded in earlier layers, using these as anchors to interpret and refine its understanding in later layers. The attention integrates information from different points in the narrative, pulling context from any position in the sequence, relying on earlier representations built up in the residual stream to maintain coherence and refine its predictions. For instance, once the model pins down `John` as the belief holder early on, it holds onto that insight as the narrative progresses, letting it shape how events are interpreted in downstream layers. This isn’t just limited to `John`—the model applies this approach across all linguistic elements, ensuring cohesive tracking throughout the sequence.

<br>

### Dictionary learning, sparse autoencoders and superposition
<sub>[↑](#top)</sub>

The linear representation hypothesis tells us that activations are **sparse**, **linear** combinations of **meaningful feature vectors**.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697598-f4d78700-4204-11eb-837c-40fb733f52df.gif" width="550px">
</p>

<br/>

Dictionary learning aligns closely with the linear representation hypothesis, aiming to express complex data as a linear combination of simpler elements, or "basis vectors". These basis vectors form a dictionary—a data structure that holds key-value pairs, when combined can efficiently represent the original data, making it easier to analyze, compress, or reconstruct. In models, a dictionary of learned concepts with associated directions allows specific elements to be activated based on relevance to the input; for example, `queen` could be represented by a combination of `female` and `royalty` directions. Sparsity is key here, as most concepts are irrelevant to a given input, resulting in many feature scores remaining zero.

Sparse autoencoders (SAEs) extend this by learning both the dictionary and a sparse vector of coefficients for each input. They're trained to reconstruct input activations, where the hidden state captures the weights of meaningful neuron combinations, and the decoder matrix learns the dictionary's feature vectors. Each latent variable in the autoencoder thus represents a distinct learned concept, enabling interpretable, causal insight into how the model organizes knowledge. SAEs leverage the hypothesis that model internals operate as sparse linear combinations of these concept directions, providing a structured way to find interpretable directions in the residual stream, MLPs, or multi-head attention.

There are many directions to find because of **1)** polysemanticity, where many neurons fire for multiple, often times unrelated features.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/16ce1f4b-32dd-486d-920b-ae2394cea058" width="950px">
</p>

<br/>

And **2)** superposition, neural networks represent more concepts (features) than they have neurons and uses linear combinations of neurons to represent these concepts. 

Basically, neurons represent multiple different things and these things are spread across multiple different neurons. Because of superposition, we have a limited number of neurons for all our features, so there are lots of features and not so many neurons in any given activation space. But the irony is that the features are actually sparse, so only a few of them are active at any given time. This allows us to take advantage of SAEs. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/e7869efd-bcaa-4d21-b49f-c0e1db1de148" width="480"/>
</p>

<br>

So we can take the activation vectors from attention, an MLP or the residual stream, expand them in a wider space using the SAE where each dimension is a new feature and the wider space will be sparse, which allows us to reconstruct the original activation vector from the wider sparse space, then we get complex features that the mechanism has learned from the input. From this we can extract rich structures and representations that the model has learned and how it views different features it has processed.

The SAE suite used is Google Deepmind's <a href="https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/" title="Google Deepmind" rel="nofollow">Gemma Scope</a>, and the output was visualized using <a href="https://docs.neuronpedia.org/" title="Neuronpedia" rel="nofollow">Neuronpedia</a>. Gemma Scope is a collection of hundreds of SAEs on every layer and sublayer of Gemma-2-2B and 9B. Using the trained SAE on the ToM passage, we can take features from layer 22 of Gemma-2-2B out of superposition, and see which features in the model are activated.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2c554f22-7de0-4b2b-9f5e-2a30faef77b3" width="480"/>
</p>

<br>

It seems like the model has specific features dedicated to representing different aspects of the narrative. For example, feature 61 focuses on *references to positions and locations in a narrative*. This feature has a high explanation score<sub>[<a href="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain" title="Bills" rel="nofollow">17</a>]</sub>, showing that the model is correctly isolating different narrative elements through distinct features.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/285680ab-c15e-46f6-9c3d-f963430fe969" width="480"/>
<img src="https://github.com/user-attachments/assets/73540c29-3935-4b85-aeff-7a2b65a738f7" width="480"/>
</p>

<br>

These features suggest that the model is building an internal representation of the physical setup described in the passage, tracking where objects and characters are placed. It’s also clear that several features are responsible for keeping track of John and Mark's movements and actions.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/5382e695-ca33-4ede-9440-461bbc902bce" width="480"/>
<img src="https://github.com/user-attachments/assets/2662e86c-7bd7-4abb-b9bc-b59033d72044" width="480"/>
<br>
<small style="font-size: 8px;">The model also has features representing actions that directly change the scene.</a></small>
</p>

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/545661b6-e6b5-46c0-b20a-5101d50c93a9" width="480"/>
</p>

<br>

One standout aspect of the model’s capacity to track scene changes lies in its approach to temporal sequencing—it’s almost like it’s keeping a detailed record of event order. Take, for instance, feature 11786, which captures *statements involving returning or coming back from a situation or event*. This kind of specialized tracking is just one of many spatial and temporal features we find scattered throughout the residual stream, indicating the model’s capability for not only for understanding static states but also for representing the flow of actions as they unfold in time and space.

The residual stream, in particular, plays a key role as an information-preservation highway across the layers. For example, it receives inputs from L10H4 and relays them through to L14H0 and then to L17H3. Through this pathway, we can observe representations of actions forming within the residual stream itself, often refined further by the MLPs.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/35024155-99d1-4595-9ed1-2ab162c7580f" width="480"/>
<img src="https://github.com/user-attachments/assets/428ae4e5-c003-4404-a124-260cc593988e" width="480"/>
</p>

<br>

In the MLP features, we're seeing a recurring theme, feature 11284 looks like it’s picking up on verbs associated with actions and states in a narrative frame. The **action related features** are a lot **clearer in the residual stream and MLPs**. This is probably helping the model track actions in the story—meanwhile, feature 5852 seems more tuned into verbs and phrases related to visual attention or perception, which may be important for encoding John’s final act of scanning the room. These features in the MLP layer are giving the model a structure for managing specific narrative events, helping it ground actions and observations.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/7479b1e5-edc3-47f8-a6e6-151a8cf0469f" width="480"/>
<img src="https://github.com/user-attachments/assets/73123c93-98f3-43c5-845d-25b3f9c7b6b8" width="480"/>
<img src="https://github.com/user-attachments/assets/4e48f9d9-1a81-472a-a21b-b058be9335c3" width="480"/>
</p>

<br>

Several features seem to be directly tied to representing belief states and knowledge. Feature 13597 is likely crucial for capturing John's lack of knowledge about what happened in the room while he was away. Feature 5107 probably signals the model’s awareness of John’s ignorance, potentially reflecting uncertainty and doubt. Feature 12703 could be involved in modeling John’s thought process when he returns to the room, helping the model represent how John updates his beliefs. These features seem important for understanding how the model processes ToM scenarios, especially when tracking characters’ evolving mental states.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/d4171c64-b7af-4ff9-83fa-36f8a4c0f03f" width="480"/>
<img src="https://github.com/user-attachments/assets/cf95b8e5-09ff-4753-8d9d-63575417fe1b" width="480"/>
<img src="https://github.com/user-attachments/assets/ac9db7c7-255a-4bf7-ae70-868f23c5a19d" width="280"/>
</p>

<br>

Another key aspect for the ToM task is spatial processing. Similar to residual stream feature 81, feature 12441 likely tracks the positions of the cat, box, and basket, while feature 14364 seems to process how subjects and objects move around the room.

What’s pretty clear from this is that the residual stream and MLP features in layer 22 show a high degree of specialization. ToM-related features are distributed across multiple distinct MLPs, suggesting the model doesn’t rely on a single "ToM module". Instead, it integrates various aspects of *reasoning* to achieve ToM understanding.

The features range from low-level tasks (like tracking object positions) to high-level abstractions (like representing uncertainty and beliefs), showing a lot of nuance. Gemma seems to have developed specialized concepts for belief representation, spatial awareness, temporal sequencing, and handling contradictory information—supplementing what we see in attention heads. It really speaks to the power of gradient descent; it’s finding solutions and representations way beyond what we’d initially predict.

<br>

### ToM circuit <a id="tom-circuit"></a> 
<sub>[↑](#top)</sub>

SAEs organize concepts into functionally coherent clusters. Because of this its possible that  LLMs might develop their own versions of brain-like regions<sub>[<a href="https://arxiv.org/html/2410.19750v1" title="Li" rel="nofollow">21</a>]</sub>. If specific attention heads are grouped into components, its possible to produce functional clusters—or subcircuits—which naturally emerge and synchronize across different positions in the input sequence.

As a rough analogue to how neural fMRI scans capture distributed activations, attention heads shift focus across tokens, similar to how brain regions activate based on focus and task demands. We can make this analogy by thinking about the parallels between functional lobes in the brain and the structure of a transformers attention mechanisms. 

Each brain lobe has a specialized role: the occipital lobe handles vision, and the frontal lobe manages planning. Attention heads work similarly, processing contextual knowledge within specific structures. Like lobes aiding decision-making by accessing relevant knowledge, attention heads enable transformers to weigh parts of the input sequence. 

If we zoom out from any single head, we can define specific attention heads across layers as circuit components. From there, we can start mapping out how these components *fire* across the ToM passage, revealing how they work together to solve the task. 

The methodology aligns closely with the original paper, but with some tweaks: activation data is collected, co-occurrence metrics are calculated, spectral clustering is applied, and affinity matrices with the Phi coefficient are used with spectral clustering. Tests were run on a small dataset that uses different templates to construct false belief passages that structurally resemble the original ToM narrative.

The results show distinct ToM subcircuits—sets of attention heads lighting up at key points during the task. These components act as cohesive units, each one relative to others, activating or staying dormant at different sequence positions. High activation levels indicate “lit-up” components compared to others. High activation values indicate components that are more activated against components that are more or less dormant.

Because of spectral clustering its possible to see which components have groups of heads that activate together across different contexts. Essentially this allows us to see how information flows through the network as its making its predictions. For example, within scene representation, certain heads may consistently activate with heads in copy suppression, particularly when managing changes in the scene and beliefs about the scene in the penultimate state. By calculating these affinities, its possible to see which specific heads within each component interact most frequently, giving insight into sub-patterns within the larger components.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/dd7d6109-c3d1-4354-a994-1d6d651fbff4" width="600"/>
</p>

<br>

We see initial-state heads co-activating with action-state heads early on, setting up a stable context for the initial and intermediate parts of the sequence with high similarity. Belief-state heads and scene-representation heads start co-activating in the intermediate and penultimate states, while still referencing that initial and action-state context—like when John or Mark leave or return. This makes sense, as the model has to keep updating its belief state about the environment based on what’s going on in the scene.

Interestingly, the initial and action states fade out as belief and scene representation heads start to integrate more of the learned context and semantics. Given the strong Q bias here, this isn’t unexpected. When John is “thinking” at the end of the sequence, “thinks” functions as a clausal complement verb, representing a mental act rather than a concrete physical or verbal action.

Copy suppression kicks in weakly at the beginning, but it steadily ramps up, progressively increasing suppression as conflicting belief states make the scene change. In the final part of the sequence, we see copy suppression co-activating with belief state heads, offsetting the model’s final prediction. It’s stopping the model from copying past states that don’t fit with the current context of where the cat is supposed to be by the end.

**NEED TO DIFFERENTIATE BETWEEN INIHIBTION AND NEGATIVE MOVER HEADS TO RELATE TO THE COPY SUPRESSION HEAD, FEELS LIKE ITS A CATCH ALL RIGHT NOW (CROSS REFERENCE WITH ACTIVATION PATCHING/PATH PATCHING/QKV PLOT RESULTS)**
**NEED TO DIFFERENTIATE BETWEEN INIHIBTION AND NEGATIVE MOVER HEADS TO RELATE TO THE COPY SUPRESSION HEAD, FEELS LIKE ITS A CATCH ALL RIGHT NOW (CROSS REFERENCE WITH ACTIVATION PATCHING/PATH PATCHING/QKV PLOT RESULTS)**
**NEED TO DIFFERENTIATE BETWEEN INIHIBTION AND NEGATIVE MOVER HEADS TO RELATE TO THE COPY SUPRESSION HEAD, FEELS LIKE ITS A CATCH ALL RIGHT NOW (CROSS REFERENCE WITH ACTIVATION PATCHING/PATH PATCHING/QKV PLOT RESULTS)**

In other words, high suppression co-activation directly affects the final predicted location of the cat. This lines up with the low activation values we’re seeing at positions connected to nouns and locations in L23H5.

It’s likely that the initial and action states act as an anchor for the subcircuit, helping it “remember” original positions and facts before any scene changes. This recurring co-activation with the initial state could be the model’s way of constantly comparing the current scene to its starting point, helping it distinguish between what’s stayed the same and what’s changed. Copy suppression’s co-activation with scene representation suggests that as the scene changes (like when `Mark moves the cat`), the model selectively downplays or retains certain facts.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/47970454-6f36-435c-befd-7b68d8c8c954" width="650"/>
<br>
<small style="font-size: 8px;">Theory of Mind Circuit.</a></small>
</p>

<br>

The pattern would suggest that the ToM circuit efficiently balances between retaining initial knowledge, updating as the story progresses, and discarding outdated beliefs or information. This aligns with human-like belief updating, where new observations modify existing beliefs without completely discarding past knowledge. It’s especially crucial for ToM, as it supports reasoning about beliefs that differ from reality—understanding what John believes (`cat on basket`) versus what is actually true (`cat on box`).

Some heads in this circuit seem to attend to previous names in the sequence but with different styles of operation. A few heads are showing a high query bias, which takes over the activation space around the basket token by focusing more on queries than keys or values. This directly impacts the belief state. Instead of nudging toward the correct prediction, these heads actually suppress the logit of the box token by writing against the belief state heads’ direction. This suppression might be doing something similar to regularization or inhibition—almost like a “negative belief state”—preventing the model from leaning too hard on certain patterns and balancing out attention across tokens.

The full circuit reveals a nuanced algorithm in its attention—and each group of heads play a distinct but interconnected role:

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
<img src="https://github.com/user-attachments/assets/1cdbad02-1845-415b-8f66-b5a120676fbd" width="650"/>
</p>

<br>

The early layers, or initial state heads, mostly handle simple linguistic elements (parts-of-speech, puncuation, determiners, conjugations, function words, syntactic dependencies) in specialized later heads.  These heads focus on picking up broader contextual signals, with key vectors usually having a larger influence. This suggests that early layers are primarily about gathering broad, diffuse information and maintaining generalized attention patterns.

As we move into the middle layers, things get more interesting. Here, the scene representation heads start doing more compositional work, integrating outputs from the initial state heads and action state heads. This is where object tracking, action understanding, and structural processing beginning to form. The attention mechanism becomes more balanced between the query and key vectors, indicating a shift towards integrating contextual information more precisely and building up a richer understanding of the scene.

This scene understanding flows into the belief state heads, especially for entities like John and Mark, where the model begins to track complex subject-object interactions and manage belief states—continuing to maintain the broader context built up the initial state, action state and scene representation heads. It’s here that we see the emergence of complex reasoning, such as tracking belief states while keeping attention on earlier elements of the narrative.

At the final stages, the copy suppression heads play a key role. These heads show both positive and negative modulations between the QK mechanisms, both enhancing and inhibiting specific connections as needed. Here, the value mechanism filters out outdated or irrelevant information, ensuring only relevant factors—like John’s incorrect belief about an object’s location—are propagated to influence the model’s final output.

So the model builds a subject's false belief about an object’s location by: **1)** Identifying John as a belief holder. **2)** Tracking the cat's movement. **3)** Updating object locations. **4)** Integrating these elements into John's belief state. **5)** Suppressing outdated or irrelevant information.

The ToM circuit satisfies the three criteria discussed in Wang et al. Minimality demonstrates each head’s contribution to ToM capability via its direct impact on logit differences by component. The score, reflecting the percentage of the model’s total logit difference (0.8365) attributed to each head, highlights the importance of each head to the task.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/aaf17afd-59ad-4002-897b-c2d29186a847" width="700"/>
<br>
<small style="font-size: 8px;"></a></small>
</p>

```markdown
Average logit difference (ToM dataset, using entire model): 0.8365
Average logit difference (ToM dataset, only using circuit): 0.8457
```

<br>

The ToM circuit hits all the key benchmarks: faithful—the circuit actually outperforms the full model slightly, showing it captures the necessary functions; complete—all heads essential for each component are included; minimal—the plot highlights clear specialization with only a minimal number of heads carrying substantial weight.

Breaking it down, the ToM circuit shows concentrated importance in certain heads, with around 35% in the scene representation heads. This suggests that understanding and keeping a coherent grasp of scene context is critical for handling ToM tasks. It implies that these heads are crucial in false belief passages, where maintaining accurate scene representations directly impacts belief tracking.

Meanwhile, the initial and action state heads contribute minimally, acting more as supporting context providers rather than the main drivers of belief tracking.

The circuit also shows a high degree of modularity: heads are highly specialized, with relevant computations neatly contained within each component. This limits interdependence with other network parts outside the defined circuit, indicating a clean and compartmentalized structure.

<br>

#### Copy supressions role in the ToM circuit

Copy supression[<a href="https://arxiv.org/pdf/2310.04625" title="McDougall" rel="nofollow">19</a>] in the ToM circuit is a head in the model that responds to the predictions that are being made by heads in earlier layers and it calibrates the final prediction. It's useful for later heads to do this because they get to see everything that comes before them. They get to see all of the context made by the earlier heads in the model and then adjust the level of confidence (positive/negative) of the next predicted token in the sequence wrt the logits before the final token is predicted.

More technically, copy surpression is an algorithm applied to the unembedding space of the model. An induction head (belief state emphasis) sees that `John put the cat on the basket`, the current token is `the` and it starts to output `basket`. This is written to the residual stream and will be mapped to the logits, but then copy supression performs post-processing on this logit space by supressing any output it has seen before that is not relevant to the induction heads context. So we can see heads that do task specific things and then heads that are responding to the previous predictions which is a more general and less specific sub task.

The amount of copy supression is mediated by the amount of attention paid to the copied token. Which makes sense, DOLMs iteratively refine their predictions by learning iteratively, being trained iteratively, then they represent information iteratively though each of its layers as information approaches the final layers.  

There's a lot more we do not know about these heads and they probably have more complex circuitry that describes when it is good to copy surpress information and when it is bad. 

(Replicate overconfidence metric analysis to test copy supression heads)

(Replicate qk, ov matrices of CS head to test its ability to produce the negative of box)

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/17c0d222-68d9-4bc5-8a7f-e2641dbe400e" width="500"/>
<br>
<small style="font-size: 8px;"></a></small>
</p>

<br>

<br>

**Provide high and low level explanation of attention heads and their patterns that make up each node in the circuit** Take all the heads from the low level data and the plots, group them in canva so see QKV patterns across each circuit component. Not sure if it'll go here, but it will be fun to see. Identify ALL copy/induction heads. Correct circuit diagram

<br>

### Ablation studies <a id="ablation-studies"></a>
<sub>[↑](#top)</sub>

Ablation studies are widely used in neuroscience and they can be applied to neural networks to assess the contribution of various components of a model to its overall performance. We systematically remove (ablate) specific components, such as neurons, layers or attention heads in the algorithm.

Mean ablating the entire ToM circuit reduces performance by ~87%.

```markdown
Original believed-actual diff: 0.836511
Ablated believed-actual diff: 0.108107
Total circuit effect: 0.728405
```

Which suggests these heads work together significantly. The remaining small difference (0.108) suggests minimal ToM capability without the circuit. Unsurprising, the most critical components are the scene representation heads and the belief state heads, where ablating reduces model performance by ~61% and ~23% respectively. No single head significantly affected performance, further validating the distributed nature of the task.

The second study tests the individual heads of the components in isolation using a baseline comparison that preserves the statistical properties of the model while ablating to measure the functional impact of the components logit difference rather than just zeroing out the activation patterns.

The point is to identify and eliminate unnecessary components to make the model more efficient. The heads with the strongest positive effects when ablated, shows performance drops (hurts performance, heads are helpful), and heads with the strongest negative effects when ablated, shows performance improves (helps performance, heads might interfere).

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/a2c9b0ee-b7ce-4714-8798-b534fc968100" width="900"/>
<br>
<small style="font-size: 8px;"></a></small>
</p>

<br>

The belief state shows ~0.5 change in logit difference when ablated from 0.8365 to ~0.3365, so we can see that L22H4 is crucial for maintaining correct belief states. Similar to scene representation L18H6 where there is a ~0.45 change. The L15H0 scene representation head shows a ~-0.2 change, which may actually interfere with belief tracking.

Given L15H0's QKV interactions, its possible that when this head is removed, others compensate by over-emphasizing belief states. Suggesting backup mechanisms possibly kick in when its ablated, or other heads might overcompensate to allow redundency in the circuit. 

#### Do statistical significance test and confidence intervals on the effect sizes

<br>

<br>

```markdown
0.8365 = Original logit diff
16.3298 = Direct Logit Attribution of top name mover head
-15.4933 = Naive prediction of post ablation logit diff
0.8365 = Logit diff after ablating L18H6
```

```markdown
Top Name Mover to ablate: 18.6
```

```markdown
Patched logit diff: 0.836511
Clean logit diff: 0.836510
Corrupted logit diff: 0.371032
Metric value: 0.000002
```



```python
Copying Scores (name
    mover heads)    
┏━━━━━━━━━┳━━━━━━━━┓
┃ Head    ┃ Score  ┃
┡━━━━━━━━━╇━━━━━━━━┩
│ (11, 5) │ 91.74% │
│ (22, 4) │ 94.50% │
│ (22, 5) │ 0.92%  │
│ Average │ 17.27% │
└─────────┴────────┘
   
   Copying Scores   
(negative name mover
       heads)       
┏━━━━━━━━━┳━━━━━━━━┓
┃ Head    ┃ Score  ┃
┡━━━━━━━━━╇━━━━━━━━┩
│ (23, 5) │ 22.94% │
│ (23, 6) │ 0.00%  │
│ Average │ 12.73% │
└─────────┴────────┘
```

<br>

# Conclusion <a id="conclusion"></a>
<sub>[↑](#top)</sub>

<br>

The results should be taken with a grain of salt, as the model was only evaluated on one ToM passage. In a future update, goal is to run a proper ablation study on multiple passages to validate or invalidate the proposed circuit.

Grab the most compelling insights from each section and reiterate on them here

The activation patching plots show that the output prediction is strongly influenced by the attention heads that focus on John's last known action and the initial state of the room. Those attention heads are..........

The circuit is not very clean, its a cyclical/recursive task

<br>

<br>

# References:
<sub>[↑](#top)</sub>

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

Li, *The Geometry of Concepts: Sparse Autoencoder Feature Structure* MIT. 2024.[<a href="https://arxiv.org/html/2410.19750v1" title="Li" rel="nofollow">21</a>]

McDougall, *Copy Suppression: Comphrehensively Understanding an Attention Head.* Independent, University of Texas, Google Deepmind. 2024.[<a href="https://arxiv.org/pdf/2310.04625" title="McDougall" rel="nofollow">22</a>]

Hardy, *Granular breakdown of data extracted from the Gemma 2-2B attention mechanism*.[<a href="https://github.com/christianThardy/christianThardy.github.io/blob/master/q-k-v-output.md" title="Hardy" rel="nofollow">23</a>]
