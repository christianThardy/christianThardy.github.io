# Deconstructing theory of mind in large language models <a id="top"></a>

<br>

##### Originally posted: 12/13/24

#### Table of Contents
- [Introduction](#introduction)
- [The Relationship Between Theory of Mind and Language](#the-relationship-between-theory-of-mind-and-language)
    - [So What?](#so-what)
- [Theory of Mind Circuit Discovery](#theory-of-mind-circuit-discovery)
    - [Principal Component Analysis](#principal-component-analysis)
    - [Identifying Relevant Layers and Activations](#identifying-relevant-layers-and-activations)
    - [Residual Stream and Multi-Head Attention](#residual-stream-and-multi-head-attention)
    - [Attention Head Analysis and Causal Tracing](#attention-head-analysis-and-causal-tracing)
    - [Dictionary Learning, Sparse Autoencoders and Superposition](#dictionary-learning-sparse-autoencoders-and-superposition)
    - [ToM Circuit](#tom-circuit)
    - [Ablation Studies](#ablation-studies)
- [Broader Implications](#broader-implications)
 - [Conclusion](#conclusion)
 - [References](#references)

<br>
 
##### tl;dr:  

*This study explores how "black box" algorithms like transformer-based large language models (LLMs) perform Theory of Mind (ToM) tasks, particularly focusing on false belief scenarios. The analysis bridges high-level behavioral analogues—such as tracking and updating belief states of entities—with low-level computational mechanisms within the model that facilitate next token prediction, to propose an algorithm that models learn to perform this task. 28 attention heads account for 16% of total heads in Gemma-2-2B and recover full ToM task performance. I'll assume you're comfortable with some basics, but I'll also be covering a lot of theory and specific technical details along the way. Feel free to hop around using the contents—if you're already familiar with most parts, you can jump straight to the results in the following sections<sub>[<a href="#conclusion" title="Go to section" rel="nofollow">1</a>]</sub><sub>[<a href="#tom-circuit" title="Go to section" rel="nofollow">2</a>]</sub><sub>[<a href="#attention-head-analysis-and-causal-tracing" title="Go to section" rel="nofollow">3</a>]</sub>.*

<br>

# Introduction

<br>

<a href="https://arxiv.org/pdf/2407.02646" title="arxiv" rel="nofollow">Mechanistic interpretability</a> gives us a way to reverse engineer the internal workings of neural networks, turning the representations they learn and the decisions they make into understandable algorithms. It's like having an x-ray or microscope to see inside the model to trace which parts of the model matter for a given task and decompose paths within the model into interpretable components that we can reason about, piece by piece.

With my current focus on transformer-based LLMs, ToM, and mechanistic interpretability, I've been wrestling with many questions about ToM tasks:

How exactly do decoder-only language models (DOLMs) perform and *solve* ToM tasks? What's happening under the hood? What kinds of algorithms is the model relying on? Is it appropriate to evaluate DOLMs the way a psychologist would analyze a human subject to gauge its level of ToM? One common framework for this is <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6974541/" title="ncbi.nlm.nih.gov" rel="nofollow">ATOMS</a> (Abilities in Theory of Mind Space), which categorizes concepts like beliefs, intentions, desires, emotions, knowledge, and percepts. Can we further contextualize this behavior by zooming in and analyzing the internal mechanisms that enable ToM capabilities in these models? 

If a DOLM is trained across multiple ToM datasets representing different categories, and has robust performance across direct probing, and we find a clear algorithmic process that leans heavily on the structure of language to solve these tasks, does that automatically mean it's not really engaging in ToM, or could it be that this is the way models represent the abstract reasoning that ToM requires? Another key question is whether ToM tasks can be solved purely by leveraging linguistic properties and syntactic structures via compositionality. 

If functional compotence (formal and social reasoning, world knowledge, situation modeling, ability to use language in real world scenarios) can be achieved from exploiting linguistic signals that represent this compositionality via formal linguistic compotence (the knowledge of grammatical and syntactic rules and statistical regularities of language)<sub>[<a href="https://arxiv.org/pdf/2301.06627" title="Mahowald" rel="nofollow">1</a>]</sub>, are these just “shortcuts” that “give answers away”<sub>[<a href="https://arxiv.org/pdf/2302.08399" title="Ullman" rel="nofollow">2</a>]</sub><sub>[<a href="https://dl.acm.org/doi/pdf/10.1145/3442188.3445922" title="Bender" rel="nofollow">3</a>]</sub>, *or* are they fundamental features that DOLMs rely on to perform and solve these tasks? 

I'm also asking myself: Do we even have a clear, interpretable algorithm for how *humans* solve ToM tasks mechanistically? Outside of the scope of combining prior knowledge with observed behaviors and contextual nuances (intentionally ignoring emotions and cultural norms) in the human brain? Neural responses are dynamic and context-dependent, as seen in how the left prefrontal cortex encodes semantic information during speech processing. It suggests that the brain uses compositionality to process language<sub>[<a href="https://www.nature.com/articles/s41586-024-07643-2" title="Jamali" rel="nofollow">4</a>]</sub>, so maybe the way models handle this through linguistic structure isn’t that far off from certain aspects of human reasoning?

There’s always the argument that model brittleness is inevitable—no dataset, no matter how large, will cover every possible scenario. New, unseen ToM data could always “break” a model. But even beyond that, do the internal mechanisms for solving this problem remain consistent across different samples? While retraining on updated datasets could lead to short-term improvements, there’s still the broader challenge of evaluating the task effectively, given both our incomplete understanding of ToM and the limitations of DOLMs.

While I'm skeptical about why models are performing ToM or are not performing ToM, I think there’s value in breaking down the abstract reasoning involved in ToM tasks into an interpretable circuit (algorithm). By understanding the internal representations in DOLMs, we can start to see how these models structure and approach ToM tasks—or more specifically false belief tasks. Even if they aren’t doing it like humans, we can still gain insights from the mechanisms they use to process “mental states”.

<br>

# The relationship between theory of mind and language

<br>

In the human brain, the language network is a set of interconnected areas in the frontal and temporal lobes that handle everything from language comprehension to generation. It's highly tuned for various linguistic operations, covering everything from word meanings (semantics) to the broader context of conversations (pragmatics).

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

Understanding and interpreting this passage to extract the meaning of each sentence requires identifying entities, their properties, and relationships—which are core tasks of semantic parsing. This would help the model comprehend the context and infer implied meanings, which is essential for making accurate ToM predictions.

ToM involves representing complex mental states and expectations. For example, in this case, DOLMs can grasp both the underlying meaning and context, allowing them to predict that `John` thinks the `cat` is on the `basket`, even though it's actually on the `box` using only formal linguistic elements. In humans this requires going beyond the literal content, and inferring the beliefs and mental states of others—key to performing these tasks.

<br>

### Pragmatics

A key concept in semantics, pragmatics focuses on how context influences the interpretation of meaning in language. This includes factors like speaker intent, conversational implicature, and situational context. To predict the final word in the example passage sequence, a model must understand not just the literal meaning of the words but also John's mental state, his expectations, and the context in which he is making the statement.

To obtain contextual understanding, we need to know the situational context—

`John` placed the `cat` on the `basket` before leaving for `school`, and he is unaware that `Mark` moved the `cat` to the `box` while he was away. 

Understanding John's beliefs and what he expects to find upon his return is crucial. We need the ability to infer the most likely location that fits John's expectation and the context (e.g. the `basket`). This involves recognizing that `John` thinks the `cat` is still where he left it, demonstrating the importance of pragmatics in interpreting language and predicting intended meaning.

<br>

## So What?
<sub>[↑](#top)</sub>

These principles and operations can *help* interpret how humans perform ToM linguistically, but how do these concepts transfer to large language models in relation to ToM? Being trained for next word prediction, LLMs end up learning a lot about the structure of language, including linguistic structures that were, until recently, thought to be out of reach for statistical models. For example, a common way to test linguistic abstraction in LLMs is through probing. This involves training a classifier on internal model representations to predict abstract categories, like part-of-speech or dependency roles. The goal is to see whether these abstract categories can be recovered from the model’s internal states. Using this method, researchers have claimed that LLMs essentially “rediscover the classical NLP pipeline”, learning linguistic features like part-of-speech tags, parse trees, and semantic roles across different layers. 

I think ToM prediction heavily relies on context to make sense of the mental states and intentions behind the words and actions of others, and final word prediction is based on implied meanings (implicature) and inferred intentions (presupposition), which are central to pragmatics. Given the literature, even if the phenomena is just statistical, some form of semantic and pragmatic inference in LLMs has been learned, regardless of how uneven or weak the performance.

One intriguing hypothesis in the psychology literature states that ToM emerges as a byproduct of learning language. The example passage above contains a sentential complement: “John thinks the cat is on the...”, and a subordinate clause: “...the cat is on the...”, nested within it. These linguistic structures are shown to play an important role in children's cognitive development, serving as a foundation for building a linguistic basis for passing false-belief tasks. In fact, *training* children on and passing complement-based language tasks is a highly significant predictor of false-belief reasoning, suggesting that adequately learning these linguistic structures may be a prerequisite for performing well on infering mental states<sub>[<a href="https://alliedhealth.ceconnection.com/files/TheRoleofLanguageinTheoryofMindDevelopment-1415277302473.pdf" title="de Villiers" rel="nofollow">5</a>]</sub>.

The study also found that children with langauge delay could not bypass the linguistic requirements of these tests by relying on alternative strategies like interpreting behavior, gestures, body cues, or life experiences. Even children with autism, provided they had sufficient language ability, required the same linguistic scaffolding to succeed. Further research corroborates this, showing that knowledge of sentential complements is a strong concurrent predictor of false-belief performance in children with autism<sub>[<a href="https://psycnet.apa.org/record/2005-12116-014" title="Tager-Flusberg" rel="nofollow">6</a>]</sub>. This aligns with the idea that a system trained to mimic humans would naturally develop ToM-like behaviors as a byproduct of learning human language. It weakly supports the hypothesis that ToM in humans may have originally developed as a side effect of increasing linguistic complexity—a fascinating example of how emergent capabilities can arise from seemingly unrelated tasks.

Thinking on a foundational level, if specific linguistic structures like sentential complements are important for the development of explicit ToM, how universal are Gricean principles (supposes that meaning transcends literal language<sub>[<a href="https://semantics.uchicago.edu/kennedy/classes/f09/semprag1/grice57.pdf" title="Grice" rel="nofollow">7</a>]</sub>) prior to attaining third degree ToM (which is most common in adolescents and adults<sub>[<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4873097/" title="Valle" rel="nofollow">8</a>]</sub>, not children around ages 1 to 4) and do the requirements for ToM definitively transcend linguistic boundaries? <a href="https://plato.stanford.edu/entries/grice/" title="standford encyclopedia of philosophy" rel="nofollow">Grice</a> suggests that conversational meaning is heavily rooted in pragmatic inference, going well beyond the literal. But if ToM hinges on particular linguistic structures, it raises an interesting tension: can the ability to generate and interpret implicatures unequivocally operate independently of formal language competence?

Expecting meaning and understanding to fully transcend linguistic boundaries ignores the empirical evidence that language proficiency is essential for some aspects of cognitive abilities. By misattributing an assumption of perfection to Grice's theories<sub>[<a href="https://www.latl.leeds.ac.uk/wp-content/uploads/sites/49/2019/05/Davies_2000.pdf" title="Davies" rel="nofollow">9</a>]</sub>, we might overestimate the extent to which meaning can be derived without reliance on specific linguistic structures.

While Grice assumes that speakers can navigate and infer meanings beyond explicit language, the reliance on specific linguistic structures for ToM development suggests that there are limits to this transcendence. In other words, the capacity to understand implied meanings (implicatures)  may not be purely universal—it might be bottlenecked by how well one can wield these linguistic tools, whether explicitly or implicitly. This perspective doesn't necessarily challenge Grice's theories but adds nuance: conversational implicature doesn’t just rely on shared norms of cooperation but also on the underlying linguistic competence of both speaker and listener. In essence, while Grice gives us a high-level roadmap for deriving implied meaning, the *implementation details*—the actual ability to pull this off—seem tied to linguistic and cognitive capabilities.

When it comes to ToM in humans, I’d hypothesize that language acts as a scaffold. Factors like age, language compotency, and executive function probably interact to determine ToM development. If this holds for humans, it might generalize to models: the number of parameters and architectural capacity could plausibly correlate with a model’s ToM performance. Are we seeing linguistic proxies for ToM emerge as models scale, or is there something deeper going on? That’s worth digging into.

<br>

# Theory of mind circuit discovery
<sub>[↑](#top)</sub>

The broader goal of this analysis is to identify the circuit responsible for modeling a false-belief task, with the more narrow focus being to pinpoint that circuit by understanding the behavior of attention heads, MLPs, and residual streams.

The model used for this analysis is Gemma-2-2B from Google's family of Gemma models. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/1efe16c2-cf0b-40a3-90df-b190f68b2960" width="250"/>
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/ecec4cba-66c3-4b05-acfc-132c66804021" width="400"/>
<br>
<small style="font-size: 12px;"><a href="https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/" title="Google Developers Blog" rel="nofollow">Google's Gemma, 2024</a></small>
</p>

<br>

It is a decoder-only transformer that has 25 layers and 7 attention heads per attention layer.

In terms of the internal mechanisms of a language model, a **feature** is a property of the input that humans can understand and is represented in the model's activations (the tokens from the ToM passage). A **circuit** informs us of how these features are extracted from the input and then processed by the model to perform specific behaviors (e.g. reasoning), which gives us an algorithmic understanding of how the model works. So first, we analyze the features, use them to trace out circuits that connect and process those features, and once we understand more circuits we can better understand the model.

To begin looking at ToM prediction through the lens of a decoder-only transformer, we can start by defining a simple hypothesis of an interpretable algorithm that focuses heavily on John’s mental state about where he placed the cat. This will serve as a starting point to understand how the model might represent and process ToM-related reasoning: 

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

And the later layers show the most spread. Progressing through the layers, it seems tokens are clustering based on functional similarities in the text. Showing clear seperation of key tokens early on (`John`, `cat`, `basket`, `box`) and having close proximity to one another in later layers, showing what could be a false belief (`John`, `cat`, `basket`).

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

Even from this limited perspective, you can see how the model is capable of distinguishing concepts, integrating contextual information, and focusing on task-relevant features in each mechanism. The differences between each mechanism highlight how they contribute to this evolving representation. Attention heads seem especially important for forming distinct, task-relevant clusters of information in deeper layers, while the pre- and post- residual stream shows how information is continuously transformed as it flows between mechanisms and layers. More on that later.

<br>

### Identifying relevant layers and activations
<sub>[↑](#top)</sub>

Thanks to <a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" title="lesswrong.com" rel="nofollow">nostalgebraist</a> we have the logit-lens —so we can track how language models refine their predictions across layers. The approach will be applied first to interpret layers and activations, and then to dive deeper into feature and circuit discovery.

This technique is essentially a causal intervention—we're directly messing with parts of the model to figure out how they contribute to the output. Most of the methods in this analysis fit this kind of framework. 

To make sense of what’s happening, we also need a solid performance metric to track how things change when we intervene. That way, we can get a clear read on how the model's behavior shifts.

For the ToM task, where the goal is to distinguish between the believed and actual locations of objects, the model needs to predict both the original and updated locations after certain actions. The metric we’ll use here is logit difference, which represents the difference between the logit of the believed location and the logit of the actual location. In this case:
`logit(basket) - logit(box)`<sub>[<a href="https://arxiv.org/pdf/2211.00593" title="Wang" rel="nofollow">10</a>]</sub>.

When we deconstruct the residual stream using the logit-lens, we can look at the residual stream after each layer and calculate the logit difference at that point. This simulates what would happen if we “deleted” all subsequent layers, giving us a snapshot of the model's evolving prediction. The final layernorm is applied to the residual stream values, which are then projected in the direction of the logit difference to measure the model's performance at each layer.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/40724e17-b54b-4d1b-aeff-0cdec72935a4" width="1700"/>
</p>

<br>

What's interesting is that the model shows almost no capacity to handle the task until we get to layer 22. And then—boom—attention layer 22 kicks in and almost all the performance happens there, and then things get a tiny bit better, then a lot worse right after layer 24. It’s not just a smooth upward trajectory; there’s a clear peak followed by a clear descent.

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

There are a couple of big meta-level takeaways here. First, even though our model has 7 attention heads in every layer, we can localize the behavior of the model to just a handful of key heads. This strongly supports the argument that attention heads are the right level of abstraction for understanding the model's behavior.

Second, the presence of negative heads is really surprising—like head 7 at layer 23, which makes the incorrect logit seven times more likely. I don’t fully understand what’s happening there yet, but it's definitely worth digging into more.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/6958f6c5-6b83-4337-ac3c-93baf669d565" width="950"/>
</p>

<br>

Looking back at the PCA output for layer 22, its clear that the model is doing something interesting in terms of concept clustering. It appears to be distinguishing between actors, objects and honing in on story elements that are crucial for ToM processing, but in a way where we can clearly see a refined heirarchical representation.

Based on the PCA, its possible that the ToM task may be aligned with the linear representation hypothesis<sub>[<a href="https://arxiv.org/pdf/2311.03658" title="Park" rel="nofollow">11</a>]</sub><sub>[<a href="https://aclanthology.org/N13-1090.pdf" title="Mikolov" rel="nofollow">12</a>]</sub> –the idea that models pick up properties of the input and represent them as directions in activation space. When we dig into layer 22's PCA, a few interesting things stand out.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/408baaa6-9596-41ac-94d1-098df04d129d" width="750"/>
</p>

<br>

The PCA breaks down into three clusters of concepts:

- Actor tokens (`John`, `Mark`, `cat`)
- Mental state tokens (`thinks`, `knows`)
- Location tokens (`basket`, `box`, `room`)

In the residual stream pre, there is clustering of scene elements and actors, and the separation between different semantic groups looks linear. But in the residual stream post (shared space where all layers interact) the separation is even clearer,  aligning these clusters more tightly around token concepts:

- `John` and `thinks`
- `basket` and initial state
- `box` and current state

The clustering remains clear as the attention and MLP layer outputs are added back to the residual stream with updated relationships. The separation of "knowledge states" (e.g. what John knows vs. doesn’t know, what Mark knows) appears linear. This makes sense because if tokens did not cluster within residual stream space, linear transformations across layers would be less informative and they wouldn't be meaningful.

Its clear that the model is keeping two separate but parallel "tracks":

- Reality track (blue): represents actual events from Mark's perspective
- Belief track (red): represents John's perspective

The key thing here is that after Mark moves the cat, the two tracks split, but the belief track stays locked into John’s original understanding. This suggests that the model is able to maintain two simultaneous yet distinct states—reality and belief—keeping them separate but interrelated to maintain parallel states. Even as the sequence progresses—Mark and John’s actions, them leaving, returning—the belief state remains consistent.

What’s also cool is that the PCA reveals these token clusters at consistently distinct distances, showing the same grouping across transformations. There’s almost a hypothetical “boundary” within the MLP and residual post layers, cleanly dividing what the model has learned about `John`, `Mark`, and their connection to the `basket`.

<br>

### Residual stream and multi-head attention
<sub>[↑](#top)</sub>

Attention heads are valuable to study because we can directly analyze their attention patterns—basically, we can see which positions they pull information from and where they move it to. This is especially helpful in our case since we're focused on the logits, meaning we can just look at the attention patterns from the final token to understand their direct impact.

One common mistake when interpreting attention patterns is to assume that the heads are paying attention to the token itself—maybe trying to account for its meaning or context. But really, all we know for sure is that attention heads move information from the residual stream at the position of that token. Especially in later layers, the residual stream might hold information that has nothing to do with the literal token at that position! For example, the period at the end of a sentence might store summary information for the entire sentence up to that point. So when a head attends to it, it’s likely moving that summary information, not caring if it ends with punctuation. This makes it hard to asses what the attention heads are doing when tokens are being attended to. 

But at the same time, I think when an attention head is attending to a token, it is accessing abstract information stored at that position.

<p align="center">
<img src="https://github.com/user-attachments/assets/31f89a77-fca5-49e1-8b52-a845ad5b2c11" width="280"/>
</p>

In transformer architectures, each token position has a residual stream—a vector that carries forward information as the model processes each layer. We can think of the residual stream as the place where everything communicated from earlier layers are communicated to later layers. It aggregates outputs from previous attention heads and MLPs—everything the model has *thought* so far.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/e59b7e99-7c6b-41cb-aeaa-84960f7a0eab" width="270"/>
</p>

Both attention heads and MLPs read from this stream, apply their edits, and then write the modified info back into the residual stream using linear operations (just simple addition). This linearity is key—it allows the input to any layer be decomposed as the sum of contributions from various mechanisms across different layers.

By the later layers, the residual stream holds rich, high-level abstractions: syntactic structures, semantic relationships, and even summaries of phrases or entire sentences. This enables the model to map syntax onto semantics in a powerful way. Attention heads read from specific positions in the residual stream and write new information to target positions, which helps move abstract, context-heavy information around—independent of specific tokens.

Going back to our period example, at the position of a period at the end of a sentence, the residual stream might hold a summary of the entire sentence rather than just the token embedding for the period itself. This layered representation is built up across attention blocks and MLPs, incorporating syntactic roles, semantic meanings, and sentence structure. Attention patterns help transfer these complex, high-level abstractions between positions, enabling the model to handle hierarchical structures.

As the model processes information, each layer can access everything from the residual stream **but focuses on specific directions** that are relevant for the task based on the similarity of information held between mechanisms. After aligning with the directions it needs, the model writes the information to another mechanism. The flow of information between mechanisms depends on how similar the directions in the residual stream are, guiding the movement of abstract information across the model. 

More on how transformers process information using linear algebra <a href="https://youtu.be/wjZofJX0v4M?si=yzNyY0gmwQ892Z6P&t=747" title="3Blue1Brown" rel="nofollow">here.</a>

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/5ddfe670-a04f-4ed6-b33c-8b164ac90691" width="280"/>
</p>

Rather than the input needing to go through every single layer of the network, the model can choose which layers it wants information to go through via the residual stream and what paths it wants to send information to. This is why we can expect model behavior to be kind of localized, so as the input goes through each mechanism, not every piece of the input will receive an activation.

The model is using the residual stream to achieve compositionality between different pieces of information. For example, there could be some attention head in layer 2 that composes with some head in layer 22. Technically this looks like some head in the 1st layer will output some vector to the residual stream, the head in the 2nd layer will take as an input the entire residual stream and mostly focus on the output of the 1st layer and run some computation on it. For any pair of composing pieces in the model, they are completely free to choose their own interpretation of the input, so there's no reason that the encoding of the information between head 0 in layer 0 and head 5 in layer 3 will be the same as the encoding between head 2 in layer 0 and head 3 in layer 1. While extremely useful, this means we can expect the residual stream to be very difficult to interpret.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/415c58d0-6975-4483-808c-31cccf887cd9" width="7500"/>  
</p>

<br>

So, what’s happening here is the model builds up hierarchical representations of language—phrases within sentences, sentences within paragraphs—and tracks sequences of events, which is particularly important for tasks like ToM, where understanding the events, the order of events, actor actions and possibly even directional or spatial information is key. 

In this framework, attention heads work like routers, directing specific pieces of information to the right places to solve the task. They aren’t just focusing on literal tokens but transferring abstract concepts like *"the last place John saw the cat"*, which aren't tied to any single token but are encoded in the residual stream.

This kind of hierarchical, nested structure in the residual stream is key to solving the ToM task. It requires the model to track what each actor knows or believes over time, which means keeping updated representations of these abstract knowledge states in the residual stream.

In any case, it’s easy to get tricked if you think an attention head is just focusing on a literal token. We should be looking at this information alongside the information stored in the residual streams at that position—which often contains abstract concepts or higher-level representations.

While keeping all of that in mind, it’s a good time to start thinking about the algorithms the model might be using when looking at some preliminary attention output. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/9a91c747-f3f6-47ad-8ecd-5124dbcbc79f"/>
<img src="https://github.com/user-attachments/assets/0492e03e-66de-49f3-af70-45918d8efc93"/>
<img src="https://github.com/user-attachments/assets/64a36cf9-5bc7-4212-ba60-08f08eb4a12a"/>
<img src="https://github.com/user-attachments/assets/f680eed9-8fe9-4636-9bd2-736f4a10424c"/>
    <small style="font-size: 12px;">Attention patterns of the heads. We can see where each token attends by the maximum value of where its attending, tokens weighted by how much information is being copied, and how much every token effects every other token.</small>
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

### Attention head analysis and causal tracing <a id="attention-head-analysis-and-causal-tracing"></a> 
<sub>[↑](#top)</sub>

To trace which parts of the model's attention are key for this task, and break down those pathways, we need a deeper dive into the attention patterns. Specifically, we want to see how the model attends to tokens related to John, his initial actions, and his final actions.

One approach is tracking the activations of key tokens (`John`, `basket`, `box`, `cat`) across layers, and showing how their representations evolve. Another approach is pinpointing which attention heads contribute most to predicting `basket`. By combining these methods we can zero in on heads that attend to both the initial state and John’s final action.

Looking at the most basic units of computation in the attentions heads will give the most fine-grained account of what is happening when the model is processing information to be sent to the MLPs. So we need to explore the roles of the query (Q), key (K), value (V), and output (O) vectors across the hierarchy of layers.

The LLMs attention mechanism will weigh the importance of different parts of the ToM passage. Each attention head computes three components:

- **Query (Q):** Determines which token positions to attend to.
- **Key (K):** Represents the tokens considered for attention at each position.
- **Value (V):** A weight determining to how relevant the key is to the query before propagating the token forward.
- **Output (O):** The information propagated forward.

The way QKV attention works is sort of like how a search engine operates. Imagine you’re looking for a video on YouTube —the text you type in the search bar is your query. The search engine then compares that query to a bunch of keys—like video titles, descriptions, tags that are stored in its database. Finally, it retrieves and ranks the best-matching videos—which are the values, and then you get the result—the output. So, attention is basically about mapping a query to the most relevant keys and pulling out the corresponding values. This allows attention heads to specialize: some heads prioritize token alignment (through Q and K), while others are focused on aggregating and relaying information (through V and O).

In somewhat technical terms, the values for the QK vectors control how much attention each token pays to others within the attention mechanism. A larger Q relative to K suggests the current token is more strongly driving the attention, meaning it's "searching" for relevant information to attend to. On the other hand, when K is larger than Q, it indicates that the token associated with K is drawing more attention from other tokens—essentially, it's being "attended to." The Vs hold the actual information or features from the input tokens and play a crucial role in determining what information is passed forward once the attention scores between Q and K are calculated.

However, it's important to note that the relative sizes of Q and K don't directly determine who is "doing the attending." Instead, both vectors interact through dot-product attention: Q represents the token initiating the attention (the one trying to find relevant content), and K represents the token being attended to (the potential source of relevant information). The attention scores are computed based on the interaction between Q and K, meaning both vectors play a role in deciding where attention is focused. The difference in their values might offer clues about the roles of specific tokens in the attention process, but both vectors contribute to the overall mechanism.

Selecting a few heads across layers, we can see how things are playing out in the context of the last token `basket` being predicted.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2f657659-d0eb-4385-98e1-fb0b984441b5" width="700"/>
<img src="https://github.com/user-attachments/assets/c4909dbf-f729-458c-9de9-b742cfdbffbf" width="700"/>
<img src="https://github.com/user-attachments/assets/d801aeb9-36f4-4ef4-95ae-958a80081bc6" width="700"/>
<br>
<small style="font-size: 12px;">From left to right let's assume: Establishing initial associations summarizing at final token position, preserving facts about the scene, preserving facts about the scene w/focus on initial context, encoding belief-related information in context & summarizing at final token position, preserving facts about the scene, tracking perspectives related to objects and actions, tracking perspective-based understanding and factual states, tracking belief states.</small>
</p>

<br>

- **Early Layers & Heads 0-10:**
    - **Layer 0.Head 7:** Attends to elements like `in` and `on` and other adpositions
    - **5.2:** Attends to `basket` and `box`, punctuation and the beginning of the sequence
    - **8.0:** Shows signs of growing attention to article-noun agreement `the cat`, `the box`, `the basket`, `the room`  
    - **10.0:** Strong focus on `is`, `on`, and `cat`, consolidating scene representation
    - **10.1:** High attention to `on` and `cat`, primarily focused on retrieving information rather than combining, Q vector spikes for subject-verb agreement `John takes`, `Mark takes`, as well as consistent attention to main verbs with minimal K activations
      <br>
    - **10.4:** Begins to differentiate between `box` and `basket` in a specialized way via prepositional phrases—`on the basket`, `off the basket`, with high activation on `the` in the last position of the sequence, indicating learned spatial relationships. Compared to head 1 in the same layer, strong V spikes for verb-object agreement (`takes the cat`, `puts it`). Highest v spikes around complete action sequences (`takes the cat and puts it on`)

- **Middle Layers & Heads 10-17:**
    - **14.0:** Attention to `basket`, `box`, and `cat`, showing clear object differentiation, increased attention to `basket`, starting to discover “belief states” (locations relevant to the position of the cat from the perspective of each actor)
    - **14.3:** Very high attention to `box`, possibly encoding the actual state
    - **14.6:** Increasing attention to `basket` compared to `box`, suggesting comparison
    - **16.0:** Focuses on `room` with moderate attention to objects and spatial relationships
    - **16.2:** Strongly attends to `box`, `basket`, and `cat`, refining object relationships
    - **16.3:** High attention to `cat` and `on`, objects and spatial relationships
    - **16.7:** Very strong, specialized attention to `box` and `basket`, and possibly comparing locations
    - **17.0:** Strong attention to `box` in its 2nd position in the sequence and `basket` at its 2nd position in the sequence, beginning of sequence -maintaining scenario context
    - **17.3:** Very high attention to `box`, possibly reinforcing where the cat is actually located
    - **17.4:** Increased focus on `basket`, and determiners beginning to emphasize the belief state
    - **17.6:** Attends to mainly determiners, especially the final one at the end of the sequence
    - **17.7:** Extremely high attention to `on`, `is`, `off` solidifying spatial relationship encoding via adpositions

<br>

We can see the model building its representation across layers, with later layers showing stronger activations for key tokens. Early to middle encodings suggest relations between grammar, spatial relationships, and initial object-subject integration. The middle to late encodings seem to refine object representations, and begin to emphasize John and Mark's belief state, then strongly maintaining those states.

We can sort of see evidence for copying heads (attend to a token and increase the probability of that token occuring again) in 0.7 and 10.1. Both showing rigid, position-based patterns, clean isolated spikes. 0.7 shows strong Q spikes at regular intervals with minimal KV interference. It might be doing token-level copying or positional tracking, but the sharp, forward, diagonal increased magnitude of Q spikes screams systematic copying with position awareness to me. 10.1 shows copy-like behavior for specific syntactic structures with regular patterns around sentence boundaries and copying verb-related information forward.

Evidence for <a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" title="Olsson" rel="nofollow">induction heads</a> (look at present token in context, look back at similar things that have happened, predict what will happen next<sub>[<a href="https://transformer-circuits.pub/2021/framework/index.html#residual-comms/" title="Elhage" rel="nofollow">13</a>]</sub>) in layer 14 head 0 and layer 17 head 3. Both showing more flexible semantic-based patterns<sub>[<a href="https://arxiv.org/pdf/2402.13055" title="Ren" rel="nofollow">14</a>]</sub>, and sharp, backwards K spikes and slight sharp forwards Q spikes. The former shows strong QK spikes at semantically similar tokens, attention to repeated patterns of actions/states, and the latter showing the tracking of recurring patterns in actor actions, and next state predictions based on previous patterns. Specifically, for the asymmetric patterns in layer 22 head 4:

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/43905290-8648-435d-820a-9526d971fe0a" width="700"/>
<br>
</p>

<br>

The highest Q attention (blue spike) is at the beginning of the sequence, around `basket` in the first mention of the basket, maybe suggesting the model is strongly querying the initial state of the room. The V attention (green) showing strong contributions around `basket`, exactly in the position where John first placed the cat, completely dominating the V attention of `box` where Mark moved the cat, and the V attention of `basket` at the beginning of the sequence.

The pattern shows the model is attending strongly to both the initial state (`cat on basket`) and the intermediate state (`cat moved to box`). The high query attention to the initial `basket` placement suggests the model understands this is relevant to John's belief state, and even captures `John` in the initial state with high attention activations relative to `Mark`. 

In the context of predicting the final token `basket`, the value contributions from both `basket` and `box` at their earlier positions in the sequence shows the model is tracking both possible locations of the cat; the real state (`cat on box`) and John's believed state (`cat on basket`), with the highest value contributions emphasizing tokens important to resolving the false belief and passing that information forward to other layers and heads. 

The strong attention to the position where John first moved the cat makes sense since that's what John last saw before leaving. The model appears to be using this head to integrate information about object locations and actor knowledge states. Given previous analysis, this head is key to some *belief state emphasis*, and likely follows a collection of heads that build up to this. 

<br>

More formally, for each token position we have QKV vectors, 

Q<sub>i</sub> K<sub>i</sub> V<sub>i</sub>

And the attention score for the tokens position to another positions,

*score*(i,j) = *softmax*((Q<sub>i</sub> · K<sub>j</sub>) / √d<sub>k</sub>)

And output for position *i* is,

out<sub>i</sub> = Σ<sub>j</sub>(*score*(i,j) × V<sub>j</sub>

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

Where the tall blue spike for `basket` is implemented via the strong Q vector weighting, which helps the model search for or focus on John's initial belief state. The strong green spikes for both `basket` and `box` positions V vectors carries location information. 

The moderate red activity combines both states, weighted by attention scores, allowing the model to maintain a strong representation of John's initial belief state of the `basket` location (false belief, contradiction), track the current state of the `box` location (true belief, reality), and weight them appropriately for belief state tracking.

In terms of linguistic representations, there are attention patterns that show action-state-verb agreements, tracking state changes through verbs. Small but consistent attention to prepositions like `on` and `off` that describe spatial relationships, which work together with the objects (`basket`/`box`) to establish location states. And there's attention around verbs that relate to mental states like `knows` and `thinks`, marking belief states. Overall it appears by this layer the model has integrated information from earlier layers and focuses on more complex contextual/semantic relationships!

In relation to this, we can also see the suppression of the actual current state (`cat on box`) in favor of the believed state (`cat on basket`). This suppression seems to primarily operate in layers 23 and 25, heads 5 and 4. So its possible these heads maintain the activation of `basket` while relatively suppressing `box`, which would preserve John's false belief about the cat's location. This can be observed in several ways:

**Attention patterns:**

- Many heads in layers 22-25 show high attention to `basket` and relatively lower attention to `box`.
- 23.5 and head 6 show particularly strong attention to `basket` over all instances of the token in the sequence, where `box` activations are relatively low.

**Activation patterns:**

- In the final layers (22-25), `basket` consistently has higher activation than `box`, despite `box` being the actual current location of the `cat`.

<br>

#### Causal Tracing: Activation patching

Activation patching is a super useful technique where internal activations in a neural network are replaced to target specific model behaviors and circuits. It allows us to choose which part to change so we can learn more about the model.

The obvious limitation of the techniques we’ve used so far is that they only focus on the final parts of the circuit—the bits that directly affect the logits—and they only show correlations at best. That’s useful, but clearly not enough to fully understand the whole circuit. What we really want is to figure out how everything composes together to produce the final output, and ideally, we’d like to build an end-to-end circuit that explains the entire behavior.

This is where activation patching comes in. Introduced in the ROME paper as causal tracing (although the history of the technique can be traced back to <a href="https://dl.acm.org/doi/pdf/10.5555/2074022.2074073" title="Pearl" rel="nofollow">Judea Pearl</a>), activation patching lets us dig deeper into the model’s internal computations. Here’s how it works:

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/055dc553-968e-42ee-80a2-76ee80902e10" width="700"/>
<br>
<small style="font-size: 12px;">Patching into a transformer can be done in a bunch of different ways (e.g. values of the residual stream, the MLP, or attention heads' output). If you want to get really granular, you can patch at specific sequence positions (not shown). This flexibility lets us explore different components of the model and figure out exactly where certain behaviors are coming from.</small>
</p>

<br>

You run the model twice—once with a *clean* input (original) that produces the correct answer, and once with a *corrupted* input (counterfactual) that doesn’t. The trick is that during the corrupted run, you intervene by patching in an activation from the clean run at a specific point in the network. Basically, you replace the corrupted activation at a certain layer and position with the corresponding clean activation and then let the model continue its computation. The key insight here is that you can measure how much this patch shifts the output toward the correct answer, we can then assess the importance of that particular activation.

By iterating over lots of different activations, you can map out which ones matter. If patching a certain activation makes a big difference in pushing the model toward the right answer, it tells us that activation is important for the task. In other words, activation patching functions as a denoising algorithm, contrasting with the noising approaches we've previously focused on. In this approach, we run the model on a corrupted input then introduce the clean input by patching in activations from the clean run. The flip side is noising, where we start with a clean input and patch in activations from the corrupted run, effectively adding noise.

With noising, just because performance drops when you ablate a component doesn’t automatically mean it was necessary for the task. For example, if you ablate layer 0 in Gemma-2-2B, performance gets much worse across a bunch of tasks, but that doesn’t mean layer 0 is specifically crucial for the ToM task. In fact, it seems to function more like an extended embedding layer—useful for processing tokens but isn’t doing anything specific to ToM. We’ll dig deeper into this later, but the key point is that noising can lead to some ambiguous results, while denoising tends to give clearer answers.

The ability to localize computations like this is a huge, if the model’s computations are spread out all over the place, it’s going to be much harder to form a clean, understandable story of what’s going on. But if we can pinpoint exactly which parts of the model matter, we can zoom in, figure out what they’re representing, how they’re connected, and ultimately have another useful tool that we can use to reverse-engineer the circuit responsible for the observed behavior.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/e9680ec6-8c8e-4afe-90a6-0b20c59c53d4" width="500">
</p>

<br/>

- 22.4 shows a large positive logit difference, indicating that this head is crucial for the final prediction of `basket`.
- There are lots of negative contributions throughout the model, but 14.3, 16.2, and 23.5 are very negative and possibly components to a supression circuit that helps the model focus on maintaining John's believed state.

An important thing to note is that these functions are not neatly isolated, but are distributed and overlapping across multiple positive and negative attention heads. For instance, several heads likely work together to represent the "mental state", and many of these heads also contribute to other tasks. Suppression-like activity, for example, doesn’t come from a single head—it emerges from the interactions between multiple heads throughout the network.

Specifically, 14.3, 16.2, 20.2, and 25.5 all show evidence of negative behavior on the final prediction. Each head has strong Q attention and low V attention to the `basket` token, and either Q or V attention to `box`. The most frequent and strongest activations are happening in the middle of the sequence.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/a424bb3e-90f7-4992-ab3f-3fd26ba45ebe" width="900">
</p>

<br/>

Diving deeper, the blue regions in this plot indicate where patching helped the model get closer to the correct prediction `basket`, red regions show where patching hurt (pushing it towards `box`), while white regions indicate neutral activations (neither positive nor negative). The clean run is the uncorrupted input—where the model gets things right (`John thinks the cat is on the basket`). The corrupted run comes from swapping adjacent tokens, which messes up the sentence’s meaning and leads to wrong answers. The goal is to patch activations from the clean run into the corrupted one at various layers and sequence positions and see how much it improves the model’s logit difference (i.e., how much closer it gets to predicting the correct answer).

Patching the `basket` token in layer 1 of the corrupted run gives a massive boost, almost recovering full performance. But, as we move to later layers, significant activation changes happen at the `the` token—which is the token right before the position of the final token, representing the model's prediction. **This shift hints at something important:** the model first focuses on where the `cat` was (`on the box`), and later on, it shifts to what word needs to be filled in (`basket` vs. `box`). There’s a super interesting pattern—starting from the `box` token in layer 0 and running up to the final `the` token in layer 25. This implies a distinct computational flow across the model’s layers. Early on, (layers 0-10) it’s all about the `box` token (likely where the model locks in the idea that the cat was on the box).

Between layers 10-20, the patching impact spreads more evenly across the key tokens. This is probably where the model’s pulling everything together, building up a complete understanding of what’s going on and learning about the `box` vs `basket` contradiction. Then, by layers 20-25, the focus shifts hard onto the final `the` token—this is where the model's deciding which word (`basket` vs. `box`) to predict. While patching `basket` is super helpful in early layers, it starts to hurt later on (negative blue regions). It seems like **the model needs to remember the cat's second position** (`box`) early on but **then "forget" it** by the end to make the right call (`basket`). This shows how the model's thinking evolves layer by layer. 

One cool takeaway is how localized the effect is—patching just a few tokens or layers can fix a lot of the model’s mistakes. It’s not spreading out the info evenly across the whole network. Instead, there’s a very directed flow of information from `box` to `the` over time, as if the relevant information for choosing `basket` over `box` is stored at the `box` token located at the position in the passage where Mark moved the cat.

**This fits with the bigger picture:** earlier layers are encoding the critical scene details (e.g., Mark moving the cat), while early and midstream activations are key for representing changes in location (whether the cat ends up on the basket or box). The whole process aligns with previous attention analyses—early layers set up the scene, mid layers handle object movement and maintaining the scene, and late layers focus on reinforcing John’s false belief.

Another takeaway is how models seem to encode and summarize abstract information at specific token positions that act as structural anchor points<sub>[<a href="https://arxiv.org/pdf/2310.15154" title="Tigges" rel="nofollow">15</a>]</sub>. Specifically, the tokens `box` and `leaves` stand out. Their isolation to patching suggests that rather than Mark or John's belief state being directly moved to the final token, these tokens seem to act as dedicated storage points—`box` representing the object’s location and `leaves` representing Mark’s action. Then the token `the` takes on a final aggregation role, pulling everything together before prediction.

Instead of always attending back to the original source tokens, the model compresses and aggregates causally relevant information at the intermediate tokens `box` and `leaves`, and passes that along to `the` at layer 22. By the time the prediction happens, all the information from earlier in the context is funneled through these positions. As a result, these tokens become just as important—if not more so—than the constituent parts of the sentence that originally introduced the information.

Weak evidence in this analysis shows that the summarization motif is not just for sentiment, but might be a general mechanism models use—in this case to track and update information about sequential events; so the model is using the tokens as a reference point to maintain a coherent representation of the scene. This behavior was discovered by patching clean residual activations for content (`box`, `leaves`) and functional (`the`) tokens into a corrupted run at specific layers to isolate their contribution.

The baseline logit difference for the clean run is 16.52. But when clean activations are patched in for `box`, `leaves` and `the`, the logit difference increases by 36%. This implies that the model is relying on these positions to store contextual information and that these tokens play a central role in the model’s predictions.

```markdown
blocks.2.hook_resid_post: Original: 16.517208099365234, Content Tokens: 119.10560607910156, 'the': 119.10560607910156
blocks.10.hook_resid_post: Original: 16.517208099365234, Content Tokens: 22.416271209716797, 'the': 22.416271209716797
blocks.22.hook_resid_post: Original: 16.517208099365234, Content Tokens: 6.501191139221191, 'the': 6.501191139221191
blocks.25.hook_resid_post: Original: 16.517208099365234, Content Tokens: 18.57611846923828, 'the': 18.57611846923828
```

We also see a sharp divergence in logit differences between the original and patched content tokens and `the` token at the end of the sequence. So early layers play a foundational role in encoding token-specific information, building up representations of individual tokens, including both semantic (content) and functional (grammatical) tokens. Divergence here reflects that removing or altering these tokens disrupts the encoding process at these layers.

```markdown
Original Logit: 16.517208099365234
Ablated Logit (Content Only): -17.1851749420166
```

Ablating the content tokens causes the logits to flip dramatically, dropping to a large negative value. Ablating the functional token `the` alongside content tokens doesn’t worsen the result. This suggests that `the`  on its own cannot contribute meaningfully to the prediction without the content tokens—its role as an anchor point seems to depend on the presence of the content tokens.

It’s plausible that the attention heads likely focus on `the` to pull in information from the content tokens, because patching `the` produces effects that closely mirror those of content tokens across layers. This aligns with the observation that patching `the` has a similar causal impact as patching content tokens—it’s not acting independently but rather facilitating the aggregation of meaningful context.

To investigate the ToM direction in the model's representation space, Distributed Alignment Search (DAS)—an optimization method that finds the best possible internal direction, by which a metric evaluates how changes of a given direction causally influence the mode's outputs—was used to identify a vector in the activation space that aligns maximally with correct predictions on the ToM task. This ToM direction was then tested for its causal role in the model predicting John’s believed location of the cat. Activations were projected onto the ToM direction and ablated, with both single and combined directional ablations used to assess its significance.

The results were striking. Ablating the ToM direction caused clear accuracy drops, highlighting its importance for belief representation. Specifically, the `box`, `leaves`, and `the` token positions at layer 22 played a critical role in task performance. Combined ablations had the most dramatic impact, causing accuracy to plummet from 0.625 pre-ablation to 0.0 post-ablation—a shocking -0.625 change. This suggests that the ToM direction, along with the specific token representations, is central to the model's ability to summarize context before making the final prediction.

This is fascinating because classical constituency theory suggests that understanding something like `the cat is on the basket` would require the model to explicitly encode a representation of `cat`. If you interfere with the model’s ability to represent `cat`, it should break down on tasks involving that idea, similarly to how intervening on tokens intermediate to the location prediction inhibits the location prediction. This principle is widely used in visual psychophysics to study encoding—you knock out specific pieces of information and see what breaks. If interfering with a representation prevents the system from performing, you’ve identified something integral. In the context of transformers, this plays out as behavioral implications of compositionality: you can test and observe how ToM directions in the residual stream encode early context and carry it forward to influence later semantics.

This lines up with the nature of ToM tasks, which require tracking both believed and actual object locations. The model appears to leverage multiple token positions (`box`, `leaves`) to maintain belief-relevant activations in parallel, processing different facets of the belief state simultaneously. There’s a clear progression: early context tokens like `box` and `leaves` store critical information, which are then funneled into the token `the` for final processing. This demonstrates a structured pipeline where information flows through specific points in the residual stream, enabling the model to piece together belief representations over time.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/cf67fbe1-e894-4118-9473-52c98f41d881" width="1000">
</p>

<br/>

The activation patching results for the breakdown between queries, keys, values and outputs shed a brighter light on what’s happening inside attention heads across layers. Let’s step back for a second—each attention head does two key things: **1)** deciding what information to move and where to attend to that information (governed by the attention pattern, controlled by the QK interaction) and **2)** deciding what information to move forward (handled by the V vectors, influenced by the OV projection). By patching either the attention pattern or the value vectors, we can tease apart which factor is more crucial and doing the heavy lifting.

In the `z` plot (output vector), patching outputs from certain heads noticeably shifts the models' output from `box` to `basket`, particularly in the last 5-10 layers. The behavior is pretty distributed, but some heads stand out: 16.7, 17.6, 22.2, 22.4 and 25.4 have the largest positive impact, along with 0.1, 3.1, 6.1, 8.1 (all previous layers have the same head position, very interesting), 12.2, 14.3, 17.3, 16.2, 20.2, and 23.5 having the largest negative impact. 

Looking at the `q` plot (Q vectors), we see familar patterns—negative heads in particular are pretty impactful, suggesting that modifying the queries’ focus is key for steering the model away from inaccurate outputs. This signal shows up across early, middle, and late layers, possibly reflecting the model’s attempts to align with the “true belief”.

The `k` plot (K vectors) is less clear, though heads like 14.1 and 17.2 show some influence. Finally, the `v` plot (V vectors) highlight a few key heads, with 22.1 and 22.2 standing out. Since Vs represent the actual information passed through attention, heads with influential Vs directly shape the model’s final predictions.

The analysis reveals that some attention heads are consistently impactful across Qs, Ks, and Vs, while others are more specialized. For instance, head 23.5 influences both Qs and outputs, while head 2 targets Ks and Vs. In layer 17, head 3’s Q plot shows a subtle negative activation shift, indicating how the model adjusts its belief about the location of the `cat`. The head assigns high activation to `box` and lower to `John`, suggesting a balance between factual grounding and perspective-taking. This adjustment becomes clearer by layer 22, head 4, where the model confidently determines box as the true location, discounting John’s outdated belief. 

My hypothesis? Qs and Ks encode separate perspectives. Qs represent the model's mental model of the cat’s location from the perspective of the actors, Ks encode the objective reality, and Vs carry the actual belief being passed forward (true or false). Zs (output) then act as the final arbiter, integrating these signals into the model’s prediction. It’s this interaction—Qs driving belief updates, Ks grounding reality, and Vs carrying the nuanced information—that nudges the model toward its final answer. It's possible to see this play out at a finer scale with causal evidence at the QKVO dimension-level, where dimensions in the attention mechanism are input tokens. Let's see if I'm wrong.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/a1f01bda-0b0b-4ce8-b6ec-51fe0d3ad4b2" width="1000">
</p>

<br/>

Since we have a high-level understanding of QKVO's, we can test whether specific subspaces in the attention mechanism are causally essential by zeroing out the top principal components in each vector—this helps isolate activation subspaces that encode *belief*-related signals at a granular level. An interesting intervention is applying a temporal ablation—starting right when Mark moves the cat—to zero out activations only after that event. If the model fails the ToM task following this intervention, it strongly suggests that the model relies on that temporal window of activations to maintain belief coherence. This provides fine-grained causal insights into how each component of the attention mechanism functions, compared to coarser-grained analyses done earlier.

To understand which features (e.g., tokens like `John`, `Mark`, `basket`, `box`) are encoded in each component, we measure how strongly each Q, K, V, or O dimension correlates with these features. For example, a particular Q-dimension might consistently activate whenever `John` appears, indicating that this query dimension is keyed to John’s perspective. A K-dimension might align with `basket`, linking that dimension to the original location of the cat. A V-dimension might respond to `cat`, encoding where and how the cat is situated at each step. By correlating these dimensions with tokens, we can infer which components carry signals about characters, actions, or locations.

Looking at 8.1:
- Q-vectors: Correlate with `John: 0.2456`, suggesting that the query signal in this head tracks John’s perspective by aligning activations with John-related input tokens.
- K-vectors: Correlate with `basket: 0.2609`, likely indexing the scene's initial location context.
- V-vectors: Correlate with `cat: -0.2666`, encoding its initial placement on the basket.
- O-dimensions: `Dim 170: 0.0885` reflects subject identity (e.g., John) and stabilizes the final representation in the output stream.

Together, the QKVO components of head 8.1 build a foundational representation of the scene from John’s perspective: the cat originally being in the basket under John’s watch.

10.5 reveals interesting dynamics:
- Q-vectors: Correlate with `Mark: 0.2651`, tracking Mark's mental representation.
- O-dimensions: `Dim 203: -0.0895` encodes subject information, while `Dim 86: -0.0665` tracks the temporal event `leaves`, anchoring contextual updates based on sequence progression.

12.2 integrates Mark’s actions:
- V-vectors: Critical state changes emerge, with `Dim 195: 0.0718` correlating with transitions and `Dim 123: 0.0337` tracking the cat’s movement.
- K-vectors: Attend to `Mark: -0.2837`, `box: -0.1342`, and `puts: 0.3218`, likely encoding Mark’s action of moving the cat from the basket to the box.

From these heads alone we can see John’s state is strongly anchored to the initial state (`basket` location), Mark’s state is more dynamic, emphasizing the current state (`box` location) and action verbs like `takes` and `puts`, suggesting the circuit models `Mark` primarily as the agent of physical change rather than maintaining a state of his beliefs.

<br>

## So What?
<sub>[↑](#top)</sub>

The model seems to have developed a systematic, multi-step process for solving this task. Demonstrating its ability track the protagonists' belief<sub>[<a href="https://arxiv.org/pdf/2302.02083" title="Kosinski" rel="nofollow">16</a>]</sub>. Early layers handle low-level tasks like syntactic dependencies, while middle layers focus on context-driven processing, identifying key facts like `cat on box`. By the time we reach the later layers, the model integrates this context and resolves ambiguities, landing on the correct conclusion (`cat on basket`) by using semantic attention patterns to disentangle competing perspectives.

### Specialization across heads

Different heads specialize in distinct functions. Take layer 22 head 4—it’s a fantastic example of specialization in action and likely represents an induction head. This head does a few key things:

**Composes and maintains perspectives:** It attends to tokens that represent an actors belief. [Check out this plot again.](https://github.com/user-attachments/assets/43905290-8648-435d-820a-9526d971fe0a) The sequence captures where John believes the cat will be located when he returns, and the heads query vectors attend to token keys that occur earlier in the sequence that match downstream patterns.

The spikes for query, key and value in this head appear concentrated on tokens earlier in the sequence, specifically in John's region where `basket` and `cat` occur with high value contributions and `box` with significantly lower value contributions, indicating these are tokens central to the repetitive patterns in the sequence. The attention seems biased toward earlier occurrences of tokens like `basket` and `cat` with stronger contributions for these earlier tokens in heads 2, 3 and 4 compared to the layers other heads, showing a clear leftward bias and the models' capability to separate John's belief from Mark's belief. 

**Resilience through sparse, localized representations:** What’s interesting is that the role the head’s take over evolves across layers. The output of a head at one layer isn’t just a simple transformation of what it did in the previous layer. There are complex interactions between heads and the residual stream, allowing the model to gradually shift its internal representation and get closer to solving the task as it moves through the layers. 

One fascinating insight is how patching just a few key components—like specific tokens or heads—with activations from a clean run is enough to steer the model back to the correct answer. This suggests the model processes information in a sparse, localized way, breaking the problem down into specialized subtasks. It doesn’t rely on a single brittle representation; instead, it layers insights, gradually refining its understanding over time. For example, the model identifies John as the belief holder early in the sequence and uses this as an anchor. 

This insight flows forward through the layers, shaping how subsequent events are interpreted. The same approach applies across the narrative—the model maintains cohesive tracking of all linguistic elements by integrating earlier representations stored in the residual stream with new information from later layers. This long-range dependency management is key to its performance.

**Sophisticated mechanisms for processing:** Zooming out, the attention head analysis shows the model has developed specialized circuits for:

- Tracking multiple states of reality: It keeps separate representations for what’s true versus what the actors believe.
- Understanding actor knowledge limitations: Heads explicitly encode what an actor knows - versus what they don’t know.
- Maintaining long-range dependencies: The model integrates information from across the sequence, ensuring coherence.
- Integrating temporal and perspective information: It distinguishes changes over time while keeping track of different viewpoints.

These capabilities allow it to handle false belief tasks by maintaining parallel representations of reality and actor knowledge states, showing sophisticated pattern matching during next-token prediction.

**Localized circuit for belief tracking:** It’s worth noting how interventions and ablation experiments reinforce the idea that these capabilities are localized (e.g. heads exhibiting induction behavior show significant performance drops when ablated).

Thinking about how the model represents the location of the cat given the data from analyzing the queries, keys, values and outputs, we can start to build a bigger, conceptual picture of what is happening. If this is our sequence where the model appears to be tracking the occurences of `box` at index 57, `basket` at indexes 18 and 29, and `the` at index 105:

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/68336ba9-9b1f-411d-8963-0daff98a26d4" width="1000">
</p>

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/33f89aa5-b44d-43f9-82e3-a1aa99819492" width="680">
</p>

<br/>

Early layers establish character roles and the initial state. Q-components highlight who is involved (John or Mark), and K/V-components encode the initial locations and items. O-components record subject identities and keep track of the scene’s initial conditions.

Middle layers integrate changes. Mark’s actions updating the cat’s location appear in the K/V vectors. Specialized subspaces (dimensions) capture transitions and timing. The network uses these signals to form two parallel representations: the true location (as Mark knows it) and the outdated location (as John believes it to be).

Late layers distinguish whose perspective the model should rely on. Some heads inhibit the influence of updated information when reasoning about John’s belief state. Others reinforce the correct perspective by emphasizing the relevant subspace dimensions that encode John’s outdated belief. A more in-depth analysis of the QKVO-dimensions can be found <a href="https://github.com/christianThardy/christianThardy.github.io/blob/master/tom-notes.md" title="ToM notes" rel="nofollow">here</a>.

The full circuit evolves from early state representations into layered belief-action integration. Each layer builds on prior patterns, maintaining Mark’s actions as current-world events while keeping John’s beliefs separate. The circuit appears to maintain a fundamental asymmetry between the two actors—highlighting a meaningful cognitive distinction. 

The system balances belief preservation and action-driven updates, forming a dual-representation architecture, tracking what Mark does to know the true state, what John believes to make the final prediction, and maintain the separation between these two representations. Copy suppression plays a crucial role here, preventing belief contamination, and enabling false-belief reasoning through a dynamic, interpretable circuit.

<br>

### Dictionary learning, sparse autoencoders and superposition
<sub>[↑](#top)</sub>

The linear representation hypothesis tells us that activations are **sparse**, **linear** combinations of **meaningful feature vectors**.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697598-f4d78700-4204-11eb-837c-40fb733f52df.gif" width="550px">
</p>

<br/>

Dictionary learning aligns closely with the linear representation hypothesis<sub>[<a href="https://transformer-circuits.pub/2023/monosemantic-features/index.html" title="Bricken" rel="nofollow">17</a>]</sub>, aiming to express complex data as a linear combination of simpler elements, or "basis vectors". These basis vectors form a dictionary—a data structure that holds key-value pairs—and when combined can efficiently represent the original data, making it easier to analyze, compress, or reconstruct. In models, a dictionary of learned concepts with associated directions allows specific elements to be activated based on relevance to the input; for example, `queen` could be represented by a combination of `female` and `royalty` directions. Sparsity is key here, as most concepts are irrelevant to a given input, resulting in many feature scores remaining zero.

Sparse autoencoders (SAEs) extend this by learning both the dictionary and a sparse vector of coefficients for each input. They're trained to reconstruct input activations, where the hidden state captures the weights of meaningful neuron combinations, and the decoder matrix learns the dictionary's feature vectors. Each latent variable in the autoencoder thus represents a distinct learned concept, enabling interpretable, causal insight into how the model organizes knowledge. SAEs leverage the hypothesis that model internals operate as sparse linear combinations of these concept directions, providing a structured way to find interpretable directions in the residual stream, MLPs, or multi-head attention.

There are many directions to find because of **1)** polysemanticity, where many neurons fire for multiple, often times unrelated features.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/b24edca2-7911-460e-a227-c6ba3434f33e" width="950px">
</p>

<br/>

And **2)** superposition, neural networks represent more concepts (features) than they have neurons and use linear combinations of neurons to represent these concepts. 

Basically, neurons represent multiple different things and these things are spread across multiple different neurons. Because of superposition, we have a limited number of neurons for all our features, so there are lots of features and not so many neurons in any given activation space. But the irony is that the features are actually sparse, so only a few of them are active at any given time. This allows us to take advantage of SAEs. 

<p align="center">
<img src="https://github.com/user-attachments/assets/4ca32983-5c7a-457c-9b29-fd01b3446650" width="480"/>
</p>

<br>

So we can take the activation vectors from attention, an MLP or the residual stream, expand them in a wider space using the SAE where each dimension is a new feature and the wider space will be sparse, which allows us to reconstruct the original activation vector from the wider sparse space, then we get complex features that the mechanism has learned from the input<sub>[<a href="https://transformer-circuits.pub/2024/scaling-monosemanticity/" title="Templeton" rel="nofollow">18</a>]</sub>. From this we can extract rich structures and representations that the model has learned.

The SAE suite used is Google Deepmind's <a href="https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/" title="Google Deepmind" rel="nofollow">Gemma Scope</a>, and the output was visualized using <a href="https://docs.neuronpedia.org/" title="Neuronpedia" rel="nofollow">Neuronpedia</a>. Gemma Scope is a collection of hundreds of SAEs on every layer and sublayer of Gemma-2-2B and 9B. Using the trained SAE on the ToM passage, we can take features from layer 22 of Gemma-2-2B out of superposition, and see which features in the model are activated.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2c554f22-7de0-4b2b-9f5e-2a30faef77b3" width="480"/>
</p>

<br>

The model has specific features dedicated to representing different aspects of the narrative in the residual stream. For example, feature 61 focuses on *references to positions and locations in a narrative*. This feature has a high explanation score<sub>[<a href="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain" title="Bills" rel="nofollow">19</a>]</sub>, showing that the model is correctly isolating different narrative elements through distinct features.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/285680ab-c15e-46f6-9c3d-f963430fe969" width="480"/>
<img src="https://github.com/user-attachments/assets/73540c29-3935-4b85-aeff-7a2b65a738f7" width="480"/>
</p>

<br>

These features suggest that the model is building an internal representation of the physical setup described in the passage, tracking where objects and actors are placed. It’s also clear that several features are responsible for keeping track of John and Mark's movements and actions.

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

The residual stream, in particular, plays a key role as an information-preservation highway across the layers. For example, it receives inputs from 10.4 and relays them through to 14.0 and then to 17.3. Through this pathway, we can observe representations of actions forming within the residual stream itself, often refined further by the MLPs.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/35024155-99d1-4595-9ed1-2ab162c7580f" width="480"/>
<img src="https://github.com/user-attachments/assets/428ae4e5-c003-4404-a124-260cc593988e" width="480"/>
</p>

<br>

In the MLP features, we're seeing a recurring theme, feature 11284 looks like it’s picking up on verbs associated with actions and states in a narrative frame. The **action related features** are a lot **clearer in the residual stream and MLPs**. This is probably helping the model track actions in the story—meanwhile, feature 5852 seems more tuned into verbs and phrases related to visual attention or perception, which may be important for encoding John’s final act of scanning the room. These features in the MLP layer are giving the model structures for managing specific narrative events, helping it ground actions and observations.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/7479b1e5-edc3-47f8-a6e6-151a8cf0469f" width="480"/>
<img src="https://github.com/user-attachments/assets/73123c93-98f3-43c5-845d-25b3f9c7b6b8" width="480"/>
<img src="https://github.com/user-attachments/assets/4e48f9d9-1a81-472a-a21b-b058be9335c3" width="480"/>
</p>

<br>

Several features seem to be directly tied to representing belief states and knowledge. Feature 13597 is likely crucial for capturing John's lack of knowledge about what happened in the room while he was away. Feature 5107 probably signals the model’s awareness of John’s ignorance, potentially reflecting uncertainty and doubt. Feature 12703 could be involved in modeling John’s thought process when he returns to the room, helping the model represent how John updates his beliefs. These features seem important for understanding how the model processes ToM scenarios, especially when tracking actors’ evolving mental states.

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

SAEs organize concepts into functionally coherent clusters. Because of this its possible that  LLMs might develop their own versions of brain-like regions<sub>[<a href="https://arxiv.org/html/2410.19750v1" title="Li" rel="nofollow">20</a>]</sub>. If specific attention heads are grouped into components, its possible to produce functional clusters—or subcircuits—which naturally emerge and synchronize across different positions in the input sequence.

As a rough analogue to how neural fMRI scans capture distributed activations, attention heads shift focus across tokens, similar to how brain regions activate based on focus and task demands. We can make this analogy by thinking about the parallels between functional lobes in the brain and the structure of a transformers attention mechanisms. Each brain lobe has a specialized role: the occipital lobe handles vision, and the frontal lobe manages planning. Attention heads work similarly, processing contextual knowledge within specific structures. Like lobes aiding decision-making by accessing relevant knowledge, attention heads enable transformers to weigh parts of the input sequence. 

If we zoom out from any single head, we can define specific attention heads across layers as circuit components. From there, we can start mapping out how these components *fire* across the ToM passage, revealing how they work together to solve the task. The methodology aligns closely with the original paper, but with some tweaks: activation data is collected, co-occurrence metrics are calculated, spectral clustering is applied, and affinity matrices with the Phi coefficient are used with spectral clustering. Tests were run on a small dataset that uses different templates to construct false belief passages that structurally resemble the original ToM narrative.

The results show distinct ToM subcircuits—sets of attention heads lighting up at key points during the task. These components act as cohesive units, each one relative to others, activating or staying dormant at different sequence positions. This makes it possible to see which components have groups of heads that activate together across different contexts, and allows us to see how information flows through the network as its making its predictions. For example, within the location state, certain heads may consistently activate with inhibition heads in suppression component, particularly when managing changes in the scene and beliefs about the scene in the penultimate state. By calculating these affinities, its possible to see which specific heads within each component interact most frequently, giving insight into sub-patterns within the larger components.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/5d325cda-9093-4db4-8bf2-505a769fefb2" width="600"/>
<br>   
<small style="font-size: 12px;">High activation values indicate components that are more activated against low activation values.</small>
</p>

<br>

We see nsubj-1 heads (John) co-activating with nsubj-2 heads (Mark) early on, setting up a stable context for the initial and intermediate parts of the sequence with higher similarity between the location state and nsubj-2 heads. Potentially suggesting its important for the model to learn the actual location of things early in the sequence.

The duplicate token heads, and to a lesser extent the induction heads of nsubj-1 co-activate throughout the sequence with high similarity to the inhibition heads, which have a negative effect on the heads for nsubj-2 up until the penultimate and final states. Showing that the model never fully disregards the actual location of the cat, but actively chooses where the cat is based on where John believes that it is.

The location state previous token heads co-activate heavily in the inital and final state. This makes sense as the model has to keep updating the belief state of John and Mark about the environment based on what’s going on in the scene throughout the sequence, its possible this specifically helps preserve initial state information for later comparison when the final location of the cat needs to be compared with beliefs. 

Inhibition heads show complementary inverse patterns with the induction heads. Strong suppression at the beginning of the sequence makes sense intuitively; the model is tracking the competing view points by nsubj-1 and nsubj-2. Inhibition's strong activations in the initial state and weaker, albeit still strong activation in the later state offsetting the model’s final prediction. 

This suggests these heads may be involved in encoding what is **NOT** true at the start by creating *negative* representations that help track what an actor doesn't “know” or “believe”. In other words, high suppression co-activation directly affects the final predicted location of the cat. This lines up with the low activation values we’re seeing at positions connected to nouns and locations in 23.5.

Induction heads show minimal activation during initial state, stronger activation during intermediate state, and the strongest activation during the final state. The pattern is suggesting that induction heads are most active when the model needs to recall and apply patterns from earlier in the sequence, particularly engaged during the final state, which is when the model needs to recall the initial state to predict John's belief and less active during initial encoding of information. So it's important for connecting later events back to earlier states.

This aligns with the QKV patterns seen from the induction head ablation studies, where these groups of heads were identified as serving distinct functions. The temporal activation patterns provide additional evidence that previous token heads serve as foundational sequential processors, and induction heads act more like specialized pattern recognition and recall mechanisms that are particularly important for handling long-range dependencies in the false belief task.

The fact that induction heads show peak activation during the final state (when the model needs to recall John's last known state) strongly suggests they play a crucial role in maintaining and retrieving relevant historical information for the false belief task.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/b3c790fe-edb3-49ac-8e13-3e6e1406811f" width="650"/>
<br>
<small style="font-size: 12px;">Theory of Mind Circuit</small>
</p>

<br>

The pattern would suggest that the ToM circuit efficiently balances between retaining initial knowledge, updating as the story progresses, and suppressing outdated information. This aligns with human-like belief updating, where new observations modify existing beliefs without completely discarding past knowledge. It’s especially crucial for ToM, as it supports reasoning about beliefs that differ from reality—understanding what John believes (`cat on basket`) versus what is actually true (`cat on box`).

Some heads in this circuit seem to attend to previous names in the sequence but with different styles of operation. A few heads are showing a high query bias, which takes over the activation space around the `basket` token by focusing more on queries than keys or values. This directly impacts the belief states. Instead of nudging toward the correct prediction, these heads actually suppress the logit of the `box` token by writing against the belief state heads’ direction. This suppression might be doing something similar to regularization or inhibition—almost like a “negative belief state”—preventing the model from leaning too hard on certain patterns and balancing out attention across tokens.

The full circuit reveals a nuanced algorithm in its attention:

- **nsubj-1 belief state (duplicate token heads)** identify early occurrences of the same tokens that represent locations, subject actions, objects and positions in relation to John.
    - e.g., cat in room, box in room, basket in room, John in room, Mark in room, John puts cat on basket
      
- **nsubj-2 belief state (duplicate token heads)** identify early occurrences of the same tokens that represent locations, subject actions, objects and positions in relation to Mark.
    - e.g., John puts cat on basket, Mark takes cat off basket, Mark puts cat on box
      
- **location state (previous-backup token heads)** captures local dependencies, primarily focusing on locations of subjects and objects with equal weight, to tokens immediately preceding the current one, placing them in the context of the ongoing scene.
    - e.g., John puts cat on basket then leaves room, Mark puts cat on box then leaves room, John returns to room, John goes to school, Mark goes to work
      
- **nsubj-1 belief state (induction heads)** captures long range dependencies, maintains the state of subjects' in the scene by detecting patterns, copying and propagating tokens forward from early tokens previous positions in the sequence.
    - e.g., John put cat on basket, John at school, John not in room, Mark not in room, cat currently on basket
      
- **Copy Supression Heads** negatively effects true-beliefs and prevents copying the actual location of the object via negative modulations from value vectors.
    - e.g., John put cat on basket, John at school, Mark takes cat off basket, Mark put cat on box, John not in room, Mark at school, cat currently on box (according to Mark's belief), cat currently on basket (according to John's belief)
        - John+++, Mark+, cat on basket++++, cat on box--
 
<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/1de75c9c-cbad-4220-92e1-23bb980748f3" width="650"/>
</p>

<br>

The early layers, or initial state heads, mostly handle simple linguistic elements (parts-of-speech, puncuation, determiners, conjugations, function words, syntactic dependencies) in specialized later heads.  These heads focus on picking up broader contextual signals, with key vectors usually having a larger influence. This suggests that early layers are primarily focused on gathering broad, diffuse information and maintaining generalized attention patterns.

As we move into the middle layers, things get more interesting. Here, the location state heads start doing more compositional work, integrating outputs from nsubj-1 state heads and nsubj-2 state heads. This is where object tracking, action understanding, and structural processing begin to take over. The attention mechanism becomes more balanced between the query and key vectors, indicating a shift towards integrating contextual information more precisely and building up a richer understanding of the scene.

This scene understanding flows into nsubj-1's induction heads, especially for entities like John and Mark, where the model begins to track complex subject-object interactions and manage belief states—continuing to maintain the broader context built up by their initial head states, and the location state heads. It’s here that we see the emergence of complex reasoning and specialized attention heads, such as tracking belief states while keeping attention on earlier elements of the narrative in relation to John.

At the final stages, the suppression heads play a key role. They show both positive and negative modulations between the QK mechanisms, enhancing and inhibiting specific connections as needed. Here, the value mechanism filters out outdated or irrelevant information to John's knowledge, ensuring only relevant factors—like John’s incorrect belief about an object’s location—are propagated to influence the model’s final output.

So the model builds the subject's false belief about an object’s location by: **1)** Identifying John as the belief holder. **2)** Tracking the cat's movement. **3)** Updating its knowlege on object locations. **4)** Integrating these elements into John's belief state. **5)** Suppressing information irrelevant to the belief holder.

The ToM circuit satisfies the three criteria discussed in Wang et al<sub>[<a href="https://arxiv.org/pdf/2211.00593" title="Wang" rel="nofollow">10</a>]</sub> . Minimality demonstrates each head’s contribution to ToM capability via its direct impact on logit differences by component. The score, reflecting the percentage of the model’s total logit difference (0.8365) attributed to each head, highlights the importance of each head to the task.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2c2fce87-170a-4f65-9d0c-c127a7ea86dd" width="700"/>
<br>
</p>

```markdown
Average logit difference (ToM dataset, using entire model): 0.8365
Average logit difference (ToM dataset, only using circuit): 0.9373
```

<br>

The ToM circuit hits all the key benchmarks: faithful—the circuit actually outperforms the full model, showing it captures the necessary functions; complete—all heads essential for each component are included; minimal—the plot highlights clear specialization with only a minimal number of heads carrying substantial weight.

Breaking it down, the ToM circuit shows concentrated importance in certain heads, with over 40% in the induction heads of nsubj-1's belief state. This suggests that understanding and keeping a coherent grasp of where John thinks the cat is, is critical for handling ToM tasks. It implies that an accurate representation of the scene from the actor who first moved the cat directly impacts belief tracking.

Meanwhile, the duplicate and previous- backup token heads contribute minimally, acting more as supporting context providers rather than the main drivers of belief tracking.

The circuit also shows a high degree of modularity: heads are highly specialized, with relevant computations neatly contained within each component. This limits interdependence with other network parts outside the defined circuit, showing a clean and compartmentalized structure.

<br>

#### Copy supressions role in the ToM circuit

Copy supression[<a href="https://arxiv.org/pdf/2310.04625" title="McDougall" rel="nofollow">21</a>] in the ToM circuit are heads in the model that respond to predictions made by prior heads and adjust the final output prediction negatively. These heads have the advantage of seeing all preceding context and intermediate predictions generated so far. By leveraging this, they can calibrate the model's confidence in predicting the next token, effectively fine-tuning the logits before the final prediction is made.

Copy surpression in later layers operates in the unembedding space of the model. Consider an induction head that's tracking the belief state. Suppose the model processes the sentence: `John put the cat on the basket`, and the current token is `the`. The induction head predicts "basket" as the next token based on the context. This prediction is written to the residual stream and will be mapped to the logits for the final output. However, before the model commits to this prediction, the copy suppression mechanism kicks in. It performs post-processing on the logits by suppressing any outputs that have been previously seen but aren't relevant to the current context established by the induction head. 

Essentially, while some heads focus on specific tasks—like predicting the next word based on the context of previous next word predictors—other heads monitor the earlier predictions and adjust them, ensuring the model doesn't over-rely on copying tokens that aren't contextually appropriate. The degree of copy suppression is influenced by how much attention the model pays to the tokens it's considering copying. This aligns with the iterative nature of LLMs. They refine their predictions layer by layer, with each layer building upon the representations from the previous ones as information flows toward the final layers. 

This is purely speculative, but I suspect the model might have the capability to represent second-order false beliefs—essentially, understanding that one person can hold a false belief about another person’s belief. This could emerge from its ability to juggle parallel representations of both true and false information, potentially through mechanisms like copy suppression.

There's a lot more we do not know about these heads and they probably have more complex circuitry that describes when it is good to surpress information and when it is bad. 

<br>

### Ablation studies <a id="ablation-studies"></a>
<sub>[↑](#top)</sub>

Ablation studies are widely used in neuroscience and they are super useful for probing neural networks as well. The idea is to systematically “remove” (or ablate) specific mechanisms—like neurons, layers, or attention heads—within the model to assess their contribution and see how much they really matter to overall performance. 

When we mean-ablate the entire ToM circuit, performance drops by about 80.66%, showing a massive reduction in the believed-actual difference and the model's confidence of the token `basket` as the correct prediction.

```markdown
Full Circuit Mean Ablation Results:
Number of heads ablated: 28
Original believed-actual diff: 0.836511
Ablated believed-actual diff: 0.162061
Total circuit effect: 0.674451
```

This suggests that these heads are working together in a highly interdependent way. The remaining performance (~16.20%) implies that outside the ToM circuit, there’s not much capacity left for correct prediction of ToM tasks, as expected. Unsurprisingly, John's duplicate token belief state heads and the copy suppression heads come out as the most critical. Ablating these reduces performance by ~14.89% and ~68.16% respectively.

<br>

## So What?
<sub>[↑](#top)</sub>

There are key interactions and patterns that we can see backed by qualitative evidence. 

Circuit components have complementary timing in the way they activate across the sequence. The location state activates early, nsubj-1 and nsubj-2 states activate more strongly in middle and later layers, showing a clear temporal progression of information processing. Components complement each other during belief processing. Belief states and inhibition head clusters show complementary patterns; one tracks beliefs, and the other tracks what's not believed. Components are processed sequentially. Previous token heads provide steady baseline processing, induction heads build up activations over the sequence, and copy suppression prevents simple copying at end.

Out of 175 total attention heads in Gemma-2-2B's attention mechanism, there are 28 that display a significant increase in ToM performance when isolated and used to solve ToM tasks, and a significant decrease in performance when they are ablated. Element-wise analysis of LLM neurons have been found to show increased firing rates for isolated sets of neurons when performing ToM tasks when compared to isolated human neurons that show consistent fire rates across similar false-belief tasks<sub>[<a href="https://arxiv.org/pdf/2309.01660" title="Jamali" rel="nofollow">22</a>]</sub>. Both cases show similar isolated activity to the attention heads identified in this study when performing ToM.

From the perspective of this task, copy suppression helps the model maintain separate representations between what is actually true (reality) and what is believed to be true (beliefs), and this has several implications for AI alignment. Because the model has learned to maintain distinct representations and track multiple potentially conflicting “versions of reality” this highlights the capability for nuanced reasoning—understanding different perspectives, and even lying. Investigating inhibition and suppression mechanisms is crucial for understanding how models might deceive, but these same capabilities could be incredibly useful for alignment research. For example, they could help with:

- Value learning: Separating “is” from “ought” to reason about values.
- Goal preservation: Keeping different types of goals or beliefs separate and coherent.
- Corrigibility: Distinguishing human beliefs from reality, and recognizing the gap between “what is” and “what should be”. 

Copy suppression could be useful to improve alignment techniques and safeguard against belief corruption. But this also raises key questions: how reliable is this mechanism for alignment? Can it scale to more complex belief systems? -What are the failure modes, especially in edge cases? These are exactly the kinds of questions we need to answer to make progress on robust alignment.

Each component serves a specific role at different points in the sequence. The timing and strength of the activations suggest a well organized circuit that tracks states, actions, beliefs using linguistic elements throughout the narrative.

<br>

# Broader implications <a id="broader-implications"></a>
<sub>[↑](#top)</sub>

A common critique of LLMs is that they rely purely on formal linguistic competence, and therefore can't truly "learn" meaning in a deep sense<sub>[<a href="https://aclanthology.org/2020.acl-main.463.pdf" title="Bender" rel="nofollow">23</a>]</sub>. However, when considering *emergent understanding*—the idea that models develop an implicit sense of meaning based on patterns in the data—It begs to question: How do mechanisms like induction heads effectively capture semantics to succeed at ToM? 

One plausible hypothesis is that while induction heads primarily track formal patterns, semantic meaning embedded in those patterns gets absorbed through training. For example, repeated references to “the cat being on the basket” provide a robust contextual anchor. Although induction heads focus on sequence-level correlations, these correlations often align with real-world semantics present in the training data. When a model predicts that “the cat is in the basket”, it might be leveraging a weakly implicit form of semantic understanding (functional competency) encoded in its layers.

This idea is particularly relevant in tasks requiring predictions about mental states or perspectives. Even if the model initially exploits high-level patterns, these patterns often align with semantic reasoning. For example, deeper layers—say, layer 22—don’t just pass through raw pattern data from earlier layers. Instead, they integrate signals representing a mix of formal linguistic structure and contextual cues. By this stage, the model might be blending formal reasoning with the semantic relationships encoded in the data.

This raises another question: When the model predicts John’s perspective in a ToM task, is it actually reasoning about John’s mental state (functional competence)? Or is it just leveraging high-level linguistic correlations (formal competence) that happen to align with correct answers? My bet is that there’s a blurry line here—meaning can emerge from form when structured, implicit grounding exists in the data.

Induction heads, though not explicitly designed to handle grounded semantics, may leverage weak forms of grounding if the training data embeds consistent patterns that correlate with meaning. For example, if the model sees “John thinks the cat is in the basket” paired repeatedly with specific outcomes, it might learn to associate those patterns with semantic relationships. By layer 22, earlier layers have already encoded semantic cues into their representations, which the deeper layers can recombine into contextually grounded predictions.

Even without explicit grounding, models trained on structured datasets can still encode weak semantic signals. Benchmarks like MMLU, ARC-C or Winogrande embed linguistic patterns that implicitly carry semantic entailments or logical structures. Models like Gemma-2-2B seem to capture these relationships effectively, even if they’re operating formally. By layer 22, relational data synthesized from earlier layers yield outputs that mimic semantic understanding.

Tasks like Winogrande make this particularly clear: While solving these tasks seems to require semantic reasoning, models often succeed by exploiting subtle textual cues embedded in the data. This suggests that while heads like 22.4 might not directly access labeled semantic relationships, they capitalize on implicit signals encoded in the training data. For example, co-occurrences of specific token patterns might encode semantic entailments without the model ever “knowing” what those entailments mean explicitly.

In large models like Gemma-2-2B, emergent semantic inference seems plausible due to the interplay between the architecture and the training data. Benchmarks like BoolQ and TriviaQA provide structured patterns that tie linguistic forms to functional outputs, creating a scaffolding for weak grounding. While induction heads and specific layers remain pattern-driven, the broader training process imbues the model with enough implicit grounding to perform tasks requiring nuanced semantic judgments. This bridges the gap between form and meaning, allowing the model to encode partial grounding—even if it never reaches full semantic understanding.

<br>

# Conclusion <a id="conclusion"></a>
<sub>[↑](#top)</sub>

By bridging high-level behavioral analogues (tracking and updating belief states of entities) with low-level computational mechanisms (transformer attention heads, MLPs and residual streams), the hope of my work here and future work is to validate or invalidate that certain heads or circuits are causally implicated in tasks that map onto ToM-like reasoning.

The proposed ToM circuit:

- Extends on the <a href="https://www.alignmentforum.org/posts/3ecs6duLmTfyra3Gp/some-lessons-learned-from-studying-indirect-object" title="alignmentforums" rel="nofollow">IOI</a> (focuses on tracking a models ability to reconstruct the syntax of natural language) work to identify specific attention heads that are pivotal to false belief tasks. The proposed circuit tracks and updates belief states of entities in regards to locations and objects using strong formal linguistic competence and tentative functional competence via the manipulation of linguistic elements, to distinguish facts from the believed reality of a 3rd person perspective.
  
    - The empirical results from activation patching identified a circuit that’s causally linked to ToM performance, and provides *some* causal evidence that form can carry function and certain heads are necessary for successful ToM-like inference. The circuit captures stable relationships (like who believes what) that go beyond surface-level token transitions. *Some* emergent semantic-like behavior exists—removing specific heads consistently reduces performance—which pushes beyond correlation towards a stronger (though still not definitive) causal story.
      
    - The fact that patching in certain Q, K, or V components from a “clean” run restores correct predictions indicates these attention heads are doing a bit more than just memorizing surface patterns. The heads appear to encode aspects of *perspective*, *belief*, and context. The strong improvements following targeted interventions suggest the model internally represents subtle cues needed for ToM tasks. This finding is still a step short of indisputable evidence for genuine semantics, but beyond naive statistical correlation.

- Is robust to targeted ablations. Critical heads responsible for ToM capabilities were isolated to validate the circuit, and the observed performance degradations and full task recovery following ablations affirm the importance of these components in maintaining robust ToM functionality.
  
    - Furthermore, experiments show that when certain tokens (those involved in representing “belief states” like where John thinks the cat is) are patched from a clean run, the corrupted model’s performance on the ToM task recovers.
      
    - Demonstrates a direct causal relationship between certain linguistic representations and ToM task performance.

- Works with copy suppression to ensure that distinct belief representations are tracked and preserved, preventing conflation between reality and differing actors' beliefs. The circuit's interplay allows for more accurate predictions of behavior based on mismatched beliefs, a hallmark of human ToM.
  
    - The removal of inhibition and induction heads impairs ToM performance. These heads ensure that “belief tokens” and “location tokens” are managed distinctly, preventing confusion between real states of the world and an agent’s belief.

- Weakly shows that as LLMs scale and learn dense correlations, they develop weak semantic grounding—patterns that mimic genuine semantic and pragmatic reasoning.
  
    - The ToM circuit appear to track particular tokens (like “basket” vs. “box”) consistently, carrying forward these representations across layers and contributing to final predictions. This suggests the model is doing more than superficial form matching; it’s maintaining stable semantic relations that resemble an understanding of the narrative.
      
    - Pretty speculative, but I think the experiments lend credence to this by pinpointing heads whose removal or modification affects semantic coherence. While it’s not definitive proof that the model truly *understands* semantics, it’s a concrete demonstration that formal pattern capturing is sufficient to manifest in behaviors associated with semantic interpretation.

The parallels to human thinking are fascinating—but still, there’s a big “but” here: how much of this translates to other model architectures and ToM tasks beyond false beliefs across a wider range of data? I think its likely other models will use similar mechanisms<sub>[<a href="https://arxiv.org/pdf/2407.10827" title="Tigges" rel="nofollow">24</a>]</sub>, but these are questions not fully answered by this work. 

Further experiments will also explore a wider range of model architectures, the application of crosscoders, transcoders, tasks, and datasets covering more ToM aspects to assess the robustness and generalizability of these findings. For example, it would be interesting to study the difference between predicting why someone will act (ex-ante—predict why a subject will perform an action.) but with more focus on explaining why they did act (ex-post—isolates how the model understands rather than being concerned with its prediction) or develop more empirical studies or experiments that directly test hypotheses about weak grounding or emergent semantics in LLMs, in a broader ToM context.

While this work aims to bring high-level behavioral understanding to how models perform ToM, there are many unanswered questions. The method the model learned to do this task, did it just memorize it? How generalizable is it? How do we know the model's methods are truly based on what it is infering will happen versus what it has memorized (developing a method to quantify this could significantly advance this debate)? It's moving linguistic elements around but does it truly understand its utterance? Concretely defined, what is *true understanding*?

Imagine someone dismisses a book’s ability to tell a story, arguing, “It’s just ink marks on paper!” Technically true, but missing the point: the magic lies in how those marks are arranged. The specific organization—words forming sentences, sentences forming a narrative—unlocks meaning, emotion, and depth. The key question isn’t whether a book is reducible to ink and paper, but whether those marks, when structured just right, can encode the rich dynamics of storytelling. Similarly, when thinking about LLMs, the question isn’t whether they’re “just matrix multiplications”, but whether their computations, when structured, can replicate the processes that underpin cognitive abilities.

I think findings between their behavior, our behavior and what's happening to them internally will get us closer to the answers. While its possible to say we have a partial map between human and machine language processing, transformers do not fully capture the consistency and generality of human cognition—they just know that if a given feature exists, another given feature is likely to come next. However, the success of the formal and functional linguistic competence of large language models should not be ignored.

<br>

# References:
<sub>[↑](#top)</sub>

Mahowald, *Dissociating Language And Thought In Large Language Models.* University of Texas at Austin, Georgia Institute of Technology, UCLA, MIT. 2024.[<a href="https://arxiv.org/pdf/2301.06627" title="Mahowald" rel="nofollow">1</a>]

Ullman, *Large Language Models Fail on Trivial Alterations to Theory-of-Mind Tasks.* Harvard. 2023.[<a href="https://arxiv.org/pdf/2302.08399" title="Ullman" rel="nofollow">2</a>]

Bender, *On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?* University of Washington, Black in AI, The Aether. 2021.[<a href="https://dl.acm.org/doi/pdf/10.1145/3442188.3445922" title="Bender" rel="nofollow">3</a>]

Jamali, *Semantic encoding during language comprehension at single-cell resolution.* Nature. 2024.[<a href="https://www.nature.com/articles/s41586-024-07643-2" title="Jamali" rel="nofollow">4</a>]

de Villiers, *The Role of Language in Theory of Mind Development.* Lippincott Williams & Wilkins. 2014.[<a href="https://alliedhealth.ceconnection.com/files/TheRoleofLanguageinTheoryofMindDevelopment-1415277302473.pdf" title="de Villiers" rel="nofollow">5</a>]

Tager-Flusberg, *How Language Facilitates the Acquisition of False-Belief Understanding in Children with Autism.* APA PsycNet. 2005.[<a href="https://psycnet.apa.org/record/2005-12116-014" title="Tager-Flusberg" rel="nofollow">6</a>] 

Grice, *Meaning.* The Philosophical Review. 1957.[<a href="https://semantics.uchicago.edu/kennedy/classes/f09/semprag1/grice57.pdf" title="Grice" rel="nofollow">7</a>] 

Valle, *Theory of Mind Development in Adolescence and Early Adulthood: The Growing Complexity of Recursive Thinking Ability.* Europe's Journal of Psychology. 2015.[<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4873097/" title="Valle" rel="nofollow">8</a>] 

Davies, *Grice’s Cooperative Principle: Getting The Meaning Across.* University of Leeds. 2015.[<a href="https://www.latl.leeds.ac.uk/wp-content/uploads/sites/49/2019/05/Davies_2000.pdf" title="Davies" rel="nofollow">9</a>] 

Wang, *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* Redwood Research, UC Berkley. 2022.[<a href="https://arxiv.org/pdf/2211.00593" title="Wang" rel="nofollow">10</a>] 

Park, *The Linear Representation Hypothesis and the Geometry of Large Language Models.* 2024.[<a href="https://arxiv.org/pdf/2311.03658" title="Park" rel="nofollow">11</a>] 

Mikolov, *Linguistic Regularities in Continuous Space Word Representations.* Microsoft Research. 2013.[<a href="https://aclanthology.org/N13-1090.pdf" title="Mikolov" rel="nofollow">12</a>]

Elhage, *A Mathematical Framework for Transformer Circuits* Anthropic. 2021.[<a href="https://transformer-circuits.pub/2021/framework/index.html#residual-comms/" title="Elhage" rel="nofollow">13</a>]

Ren, *Identifying Semantic Induction Heads to Understand In-Context Learning* Shanghai Jiao Tong University, Shanghai Artificial Intelligence Laboratory, Fudan University, The Chinese University of Hong Kong. 2024.[<a href="https://arxiv.org/pdf/2402.13055" title="Ren" rel="nofollow">14</a>]

Tigges, *Linear Representation of Sentiment in Large Language Models* Eluether AI Institute, SERI MATS, Stanford University, Pr(AI)R Group, Independent. 2023.[<a href="https://arxiv.org/pdf/2310.15154" title="Tigges" rel="nofollow">15</a>]

Kosinski, *Evaluating Large Language Models in Theory of Mind Tasks.* Stanford University. 2023.[<a href="https://arxiv.org/pdf/2302.02083" title="Kosinski" rel="nofollow">16</a>]

Bricken, *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* Anthropic. 2023.[<a href="https://transformer-circuits.pub/2023/monosemantic-features/index.html" title="Bricken" rel="nofollow">17</a>]

Templeton, *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Anthropic. 2024.[<a href="https://transformer-circuits.pub/2024/scaling-monosemanticity/" title="Templeton" rel="nofollow">18</a>]

Bills, *Language models can explain neurons in language models* OpenAI. 2023.[<a href="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain" title="Bills" rel="nofollow">19</a>]

Li, *The Geometry of Concepts: Sparse Autoencoder Feature Structure* MIT. 2024.[<a href="https://arxiv.org/html/2410.19750v1" title="Li" rel="nofollow">20</a>]

McDougall, *Copy Suppression: Comphrehensively Understanding an Attention Head.* Independent, University of Texas, Google Deepmind. 2023.[<a href="https://arxiv.org/pdf/2310.04625" title="McDougall" rel="nofollow">21</a>]

Jamali, *Unveiling theory of mind in large language models: A parallel to single neurons in the human brain.* Massachusetts General Hospital, Harvard Medical School, MIT. 2023.[<a href="https://arxiv.org/pdf/2309.01660" title="Jamali" rel="nofollow">22</a>]

Bender, *Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data.* University of Washington, Saarland University. 2020.[<a href="https://aclanthology.org/2020.acl-main.463.pdf" title="Bender" rel="nofollow">23</a>]

Tigges, *LLM Circuit Analyses Are Consistent Across Training and Scale.* EleutherAI, ILLC, University of Amsterdam, Brown University. 2024.[<a href="https://arxiv.org/pdf/2407.10827" title="Tigges" rel="nofollow">24</a>]

McDougall, *Indirect Object Identification Exercises and Solutions used in sections 3.2, 3.3, 3.4* Independent. 2024.[<a href="https://colab.research.google.com/drive/1KgrEwvCKdX-8DQ1uSiIuxwIiwzJuQ3Gw?usp=sharing#scrollTo=IzLtmTaNl6mM5" title="McDougall" rel="nofollow">25</a>]

Hardy, *Granular breakdown of data extracted from the Gemma 2-2B attention mechanism explained by ChatGPT-4o*.[<a href="https://github.com/christianThardy/christianThardy.github.io/blob/master/qkv-output.md" title="Hardy" rel="nofollow">26</a>]

Hardy, *Code for the project can be found here*.[<a href="" title="Hardy" rel="nofollow">27</a>]
