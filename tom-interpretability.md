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

*This study explores how transformer-based large language models (LLMs) perform Theory of Mind (ToM) tasks, particularly focusing on false belief scenarios. The analysis bridges high-level behavioral analogues—such as tracking and updating belief states of entities—with low-level computational mechanisms within the model that facilitate next token prediction, to propose an algorithm that models learn to perform this task. A circuit of 28 attention heads account for 16% of total heads in Gemma-2-2B and recover full ToM task performance. I'll assume you're comfortable with some basics, but I'll also be covering a lot of theory and specific technical details along the way. Feel free to hop around using the contents—if you're already familiar with most parts, you can jump straight to the results in the following sections<sub>[<a href="#conclusion" title="Go to section" rel="nofollow">1</a>]</sub><sub>[<a href="#tom-circuit" title="Go to section" rel="nofollow">2</a>]</sub><sub>[<a href="#attention-head-analysis-and-causal-tracing" title="Go to section" rel="nofollow">3</a>]</sub>.*

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

Looking at the most basic units of computation in the attention heads will give the most fine-grained account of what is happening when the model is processing information to be sent to the MLPs. So we need to explore the roles of the query (Q), key (K), value (V), and output (O) vectors across the hierarchy of layers.

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
<small style="font-size: 12px;">From left to right: Representations generalizing then specializing across layers.</small>
</p>

<br>

The model builds its representation across layers, with later layers showing stronger activations for key tokens. Analyzing the activation contribution across tokens, it looks like early to middle encodings suggest relations between grammar, spatial relationships, and initial object-subject integration. The middle to late encodings seem to refine object representations, and begin to emphasize John and Mark's state of the scene, then strongly maintaining those states.

We can sort of see evidence for copying heads (attend to a token and increase the probability of that token occuring again) in 0.7 and 8.0. Both showing rigid, position-based patterns, clean isolated spikes. 0.7 shows strong Q spikes at regular intervals with minimal KV interference. It might be doing token-level copying or positional tracking, but the sharp, forward, diagonal increased magnitude of Q spikes screams systematic copying with position awareness to me. 8.0 shows copy-like behavior for specific syntactic structures with regular patterns around sentence boundaries and copying verb-related information forward.

Evidence for <a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" title="Olsson" rel="nofollow">induction heads</a> (look at present token in context, look back at similar things that have happened, predict what will happen next<sub>[<a href="https://transformer-circuits.pub/2021/framework/index.html#residual-comms/" title="Elhage" rel="nofollow">13</a>]</sub>) in layer 14 head 2 and layer 17 head 0. Both showing more flexible semantic-based patterns<sub>[<a href="https://arxiv.org/pdf/2402.13055" title="Ren" rel="nofollow">14</a>]</sub>, and sharp, backwards K spikes and slight sharp forwards Q spikes. The former shows strong QK spikes at semantically similar tokens, attention to repeated patterns of actions/states, and the latter showing the tracking of recurring patterns in subject actions, and next state predictions based on previous patterns. Specifically, for the asymmetric patterns in layer 22 head 4:

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/43905290-8648-435d-820a-9526d971fe0a" width="700"/>
<br>
</p>

<br>

The pattern shows the model is attending strongly to both the initial state (`cat on basket`) and the intermediate state (`cat moved to box`). The high query attention to the initial `basket` placement suggests the model understands this is relevant to John's belief state, and even captures `John` in the initial state with high attention activations relative to `Mark`. 

In the context of predicting the final token `basket`, the value contributions from both `basket` and `box` at their earlier positions in the sequence shows the model is tracking both possible locations of the cat; the real state (`cat on box`) and John's believed state (`cat on basket`), with the highest value contributions emphasizing tokens important to resolving the false belief and passing that information forward to other layers and heads. 

The strong attention to the position where John first moved the cat makes sense since that's what John last saw before leaving. The model appears to be using this head to integrate information about object locations and subject knowledge states. Given previous analysis, whether this head is an induction head or not, its key to some *belief state emphasis*, and likely follows a collection of heads that build up to this. 

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

In terms of linguistic representations, there are attention patterns that show action-state-verb agreements, tracking state changes through verbs. Small but consistent attention to prepositions like `on` and `off` that describe spatial relationships, which work together with the objects (`basket`/`box`) to establish location states. There's attention around verbs that relate to mental states like `knows` and `thinks`, marking belief states.

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

This is where causal tracing comes in. First introduced in the ROME paper (although the history of the technique can be traced back to <a href="https://dl.acm.org/doi/pdf/10.5555/2074022.2074073" title="Pearl" rel="nofollow">Judea Pearl</a>), activation patching lets us dig deeper into the model’s internal computations. Here’s how it works:

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/055dc553-968e-42ee-80a2-76ee80902e10" width="700"/>
<br>
<small style="font-size: 12px;">Patching into a transformer can be done in a bunch of different ways (e.g. values of the residual stream, the MLP, or attention heads' output). If you want to get really granular, you can patch at specific sequence positions (not shown). This flexibility lets us explore different components of the model and figure out exactly where certain behaviors are coming from.</small>
</p>

<br>

You run the model twice—once with a *clean* input (original) that produces the correct answer, and once with a *corrupted* input (counterfactual) that doesn’t. The trick is that during the corrupted run, you intervene by patching in an activation from the clean run at a specific point in the network. Basically, you replace the corrupted activation at a certain layer and position with the corresponding clean activation and then let the model continue its computation. The key insight here is that you can measure how much this patch shifts the output toward the correct answer, we can then assess the importance of that particular activation.

By iterating over lots of different activations, you can map out which ones matter. If patching a certain activation makes a big difference in pushing the model toward the right answer, it tells us that activation is important for the task. In other words, activation patching functions as a denoising algorithm. In this approach, we run the model on a corrupted input then introduce the clean input by patching in activations from the clean run. The flip side is noising, where we start with a clean input and patch in activations from the corrupted run, effectively adding noise.

With noising, just because performance drops when you ablate a component doesn’t automatically mean it was necessary for the task. For example, if you ablate layer 0 in Gemma-2-2B, performance gets much worse across a bunch of tasks, but that doesn’t mean layer 0 is specifically crucial for the ToM task, it seems to function more like an extended embedding layer which is useful for processing tokens but isn’t doing anything specific to ToM. The key point is that noising can lead to some ambiguous results, while denoising tends to give clearer answers.

The ability to localize computations like this is huge, if the model’s computations are spread out all over the place, it’s going to be much harder to form a clean, understandable story of what’s going on. But if we can pinpoint exactly which parts of the model matter, we can zoom in, figure out what they’re representing, how they’re connected, and ultimately have another useful tool that we can use to reverse-engineer the circuit responsible for the observed behavior.

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

Diving deeper, the blue regions in this plot indicate where patching helped the model get closer to the correct prediction `basket`, red regions show where patching hurt (pushing it towards `box`), while white regions indicate neutral activations (neither positive nor negative, transplanting the clean run into the model has no effect). The clean run is the uncorrupted input—where the model gets things right (`John thinks the cat is on the basket`). The corrupted run comes from swapping adjacent tokens, which messes up the meaning of the sentence and leads to wrong answers. The goal is to patch activations from the clean run into the corrupted one at various layers and sequence positions and see how much it improves the model’s logit difference (i.e., how much closer it gets to predicting the correct answer).

Patching the `basket` token in layer 1 of the corrupted run gives a massive boost, almost recovering full performance. But, as we move to later layers, significant activation changes happen at the `the` token—which is the token right before the position of the final token, representing the model's prediction. **This shift hints at something important:** the model first focuses on where the `cat` was (`on the box`), and later on, it shifts to what word needs to be filled in (`basket` vs. `box`). There’s a super interesting pattern—starting from the `box` token in layer 0 and running up to the final `the` token in layer 25. This implies a distinct computational flow across the model’s layers. Early on, (layers 0-10) it’s all about the `box` token (likely where the model locks in the idea that the cat was on the box).

Between layers 10-15, the patching impact spreads more evenly across the key tokens. This is probably where the model’s pulling everything together, building up a complete understanding of what’s going on and learning about the `box` vs `basket` contradiction. Then, by layers 20-25, the focus shifts hard onto the `leaves` token and the final `the` token—this is where the model's deciding which word (`basket` vs. `box`) to predict. While patching `basket` is super helpful in early layers, it starts to hurt later on (negative blue regions). It seems like **the model needs to remember the cat's second position** (`box`) early on but **then "forget" it** by the end to make the right call (`basket`). This shows how the model's thinking evolves layer by layer. 

One cool takeaway is how localized the effect is—patching just a few tokens or layers can fix a lot of the model’s mistakes. It’s not spreading out the info evenly across the whole network. Instead, there’s a very directed flow of information from `box` to `the` over time, as if the relevant information for choosing `basket` over `box` is stored at the `box` token located at the position in the passage where Mark moved the cat.

**This fits with the bigger picture:** earlier layers are encoding the critical scene details (e.g., Mark moving the cat), while early and midstream activations are key for representing changes in location (whether the cat ends up on the basket or box). The whole process aligns with previous attention analyses—early layers set up the scene, mid layers handle object movement and maintaining the scene, and late layers focus on reinforcing John’s false belief.

Another takeaway is how models seem to encode and summarize abstract information at specific token positions that act as structural anchor points<sub>[<a href="https://arxiv.org/pdf/2310.15154" title="Tigges" rel="nofollow">15</a>]</sub>. Specifically, the tokens `box` and `leaves` stand out. Their isolation to patching suggests that rather than Mark or John's belief state being directly moved to the final token, or attending back to the original source tokens,  the model compresses and aggregates causally relevant information at the intermediate tokens `box` and `leaves` in the residual stream—`box` representing the object’s location and `leaves` representing Mark’s action. 

Then the token `the` takes on a final aggregation role, pulling everything together before prediction at layer 22. By the time the prediction happens, all the information from earlier in the context is funneled through these “storage” positions. As a result, these tokens become just as important—if not more so—than the constituent parts of the sentence that originally introduced the information.

Weak evidence in this analysis shows that the summarization motif may not just be for sentiment, but might be a general mechanism models use to track and update information about sequential events; so the model is using the tokens as a reference point to maintain a coherent representation of the scene—in this case for ToM. This behavior was discovered by patching clean residual activations for content (`box`, `leaves`) and functional (`the`) tokens into a corrupted run at specific layers to isolate their contribution.

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

Ablating the ToM direction caused clear accuracy drops, highlighting its importance for belief representation. Specifically, the `box`, `leaves`, and `the` token positions at layer 22 played a critical role in task performance. Combined ablations had the most dramatic impact, causing accuracy to plummet from 0.625 pre-ablation to 0.0 post-ablation—a shocking -0.625 change. This suggests that the ToM direction, along with the specific token representations, is central to the model's ability to summarize context before making the final prediction.

This is fascinating because classical constituency theory suggests that understanding something like `the cat is on the basket` would require the model to explicitly encode a representation of `cat`. If you interfere with the model’s ability to represent `cat`, it should break down on tasks involving that idea, similarly to how intervening on tokens intermediate to the location prediction inhibits the prediction. This principle is widely used in visual psychophysics to study encoding—you knock out specific pieces of information and see what breaks. 

If interfering with a representation prevents the system from performing, you’ve identified something integral. In the context of transformers, this plays out as behavioral implications of compositionality: you can test and observe how ToM directions in the residual stream encode early context and carry it forward to influence later semantics.

This lines up with the nature of ToM tasks, which require tracking both believed and actual object locations. The model appears to leverage multiple token positions (`box`, `leaves`) to maintain belief-relevant activations in parallel, processing different facets of the belief state simultaneously. There’s a clear progression: early context tokens like `box` and `leaves` store critical information, which are then funneled into the token `the` for final processing. This demonstrates a funky, structured, memory-like pipeline where information flows through specific points in the residual stream, enabling the model to piece together belief-related representations over time.

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

My hypothesis? Qs and Ks encode separate perspectives. Qs represent the model's mental model of the cat’s location from the perspective of the subjects, Ks encode the objective reality, and Vs carry the actual belief being passed forward (true or false). Zs (output) then act as the final arbiter, integrating these signals into the model’s prediction. It’s this interaction—Qs driving belief updates, Ks grounding reality, and Vs carrying the nuanced information—that nudges the model toward its final answer. It's possible to see this play out at a finer scale with causal evidence at the QKVO dimension-level, where dimensions in the attention mechanism are input tokens.

<br>

#### Causal Tracing: Path patching

How might the behavior of a model change if we selectively replace the output of attention head A directed toward head B (where B follows A in the computation sequence) with the corresponding value from a different input distribution, while keeping all other components unchanged? What if we do this across different head types? Path patching will shift the focus from evaluating the isolated importance of individual attention heads to understanding the functional role of the circuit formed by their connection.

This causal intervention captures the complex interdependencies between attention heads and shows how the model's circuitry works together to solve the ToM task. The experiment will define attention head groups (e.g., “previous token heads”, “induction heads”) identified by a set of metrics that determine whether a model's attention head is acting like a specific head from the head group<sub>[<a href="https://arxiv.org/pdf/2407.10827" title="Tigges" rel="nofollow">16</a>]</sub>. Multiple path patching experiments are run to compute the clean and corrupted logits, the activations from heads that send information into receiving heads are patched, and the logit difference is measured to calculate the impact on model output.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/f0040c73-b524-4279-96cd-d6b7492f4bf8" width="500">
</p>

<br/>

After this process we have a couple artifacts. The first one is a plot that shows sender-receiver pairs (y and x axis) that shows us how things flow through the network to reveal the circuits structure. A positive effect of the magnitude of the influence of heads means that patching the sender’s activation in the receiver context tends to increase the difference between correct and incorrect logits (improving correctness or pushing in some direction), while negative values push in the opposite direction (blue). Each cell represents how much patching the activation from a sender head to a receiver head affects the model's performance.

The idea is that you take the activations from a “sender head” in the corrupted scenario and insert them into the clean scenario model run at the same point, effectively asking: “How does changing what this one head writes cause changes to the final output and to other heads downstream?”.

The darkest blue patches appear when 5.4, a previous token head, is the sender, and 12.3, a previous token head, is the receiver.

```markdown
Layer 5 Head 4 → Layer 12 Head 3
effect_of_head_to_head: -1.3715
```

Suggesting 12.3 is particularly sensitive to interference from other heads. Many of the strongest positive effects are around a value of 0.3 to 0.5, where the highest values tend to appear in interactions between later layers (L17-L23), although the receiver 12.2 shows very strong positive interactions (red) with itself and middle layers, long with 17.6 (induction head).

```markdown
Layer 12 Head 2 → Layer 12 Head 2
effect_of_head_to_head: 0.5152

Layer 17 Head 6 → Layer 17 Head 6
effect_of_head_to_head: 0.4579
```

It appears that certain heads, particularly in layers 8 and 12, are critical junction points in the network, while later layers (especially around layers 17 and 22) are important for positive reinforcement of the model's computations. But why? We get the second artifact by combining this to the flow of QKVO attention weights between heads, then we will be able to see how the identified heads form a layered composition.

Since we have a high-level understanding of the information flow between heads, we can examine how each attention head interacts and influences each other, tracking the flow of information to see how one head (sender) influences another (receiver).

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/fad49b2d-3333-439c-84cb-abeef8542d96" width="500">
<br>
<small style="font-size: 12px;">Dozens of heatmaps corresponding to all QKVO compositions in the collection of identified attention heads.</small>
</p>

<br/>

By extracting the Q, K, V, and O vectors for any head/layer, we can visualize specific compositions to analyze:

- Q composition: How the receiver's queries attend to the sender's outputs
- K composition: How the receiver's keys interact with the sender's outputs
- V composition: How the receiver's values are influenced by the sender's outputs
- O composition: How the final outputs compose between heads

We can then understand which features—tokens like `John`, `Mark`, `basket`, `box`—are encoded in each component, and measure how strongly each dimension correlates with these features. This provides a fine-grained view into how each component of the attention mechanism functions, compared to coarser-grained analyses done earlier.

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/a0579eaa-09bb-4920-8a1b-4295f07ad731" width="500">
</p>

<br/>

For example, a particular K-dimension receiving attention weights from an O-dimension might consistently activate whenever `box` appears, indicating that this output dimension is keyed to John’s perspective. A Q-dimension might align with `basket`, linking that dimension to the original location of the cat. A V-dimension might respond to `cat`, encoding where and how the cat is situated at each step. By correlating these dimensions with the corresponding tokens, we can infer which components carry signals about characters, actions, or locations between heads.

In this particular heatmap, where 5.4 is the sender, 8.1 is the receiver, and the keys of 8.1 are attending to the output of 5.4, we can see duplicate tokens aligning with high strength, with the attention weights biasing Mark's perspective of where he moves the cat.

<br>

## So What?
<sub>[↑](#top)</sub>

The model seems to have developed a systematic, multi-step process for solving this task. Demonstrating its ability track the protagonists' belief<sub>[<a href="https://arxiv.org/pdf/2302.02083" title="Kosinski" rel="nofollow">17</a>]</sub>. Early layers handle low-level tasks like syntactic dependencies, while middle layers focus on context-driven processing, identifying key facts like `cat on box`. By the time we reach the later layers, the model integrates this context and resolves ambiguities, landing on the correct conclusion (`cat on basket`) by using semantic attention patterns to disentangle competing perspectives.

### Specialization across heads

Different heads specialize in distinct functions. Take layer 22 head 4—it’s a fantastic, likely example of specialization in action. This head does a few key things:

**Composes and maintains perspectives:** It attends to tokens that represent the subject's belief. [Check out this plot again.](https://github.com/user-attachments/assets/43905290-8648-435d-820a-9526d971fe0a) The sequence captures where John believes the cat will be located when he returns, and the head's query vectors attend to token keys that occur earlier in the sequence that match downstream patterns.

The spikes for query, key and value in this head appear concentrated on tokens earlier in the sequence, specifically in John's region where `basket` and `cat` occur with high value contributions and `box` with significantly lower value contributions, indicating these tokens are central to the repetitive patterns in the sequence. The attention seems biased toward earlier occurrences of tokens like `basket` and `cat` with stronger contributions for these earlier tokens in heads 2, 3 and 4 compared to the layers other heads, showing a clear leftward bias and the models' capability to separate John's belief from Mark's belief. 

**Resilience through sparse, localized representations:** What’s interesting is that the role the head’s take over evolves across layers. The output of a head at one layer isn’t just a simple transformation of what it did in the previous layer. There are complex interactions between heads and the residual stream, allowing the model to gradually shift its internal representation and get closer to solving the task as it moves through the layers. 

One fascinating insight is how patching just a few key components—like specific tokens or heads—with activations from a clean run is enough to steer the model back to the correct answer. This suggests the model processes information in a sparse, localized way, breaking the problem down into specialized subtasks. It doesn’t rely on a single brittle representation; instead, it layers insights, gradually refining its understanding over time. For example, the model identifies John as the belief holder early in the sequence and uses this as an anchor. 

This insight flows forward through the layers, shaping how subsequent events are interpreted. The same approach applies across the narrative—the model maintains cohesive tracking of all linguistic elements by integrating earlier representations stored in the residual stream with new information from later layers. This long-range dependency management is key to its performance.

**Sophisticated mechanisms for processing:** Zooming out, the attention head analysis shows the model has developed specialized circuits for:

- Tracking multiple states of reality: It keeps separate representations for what’s true versus what the subjects believe.
- Understanding subject knowledge limitations: Heads explicitly encode what a subject knows - versus what they don’t know.
- Maintaining long-range dependencies: The model integrates information from across the sequence, ensuring coherence.
- Integrating temporal and perspective information: It distinguishes changes over time while keeping track of different viewpoints.

These capabilities allow it to handle false belief tasks by maintaining parallel representations of reality and subject knowledge states, showing sophisticated pattern matching during next-token prediction.

**Localized circuit for belief tracking:** It’s worth noting how interventions and ablation experiments reinforce the idea that these capabilities are localized (e.g. heads exhibiting induction behavior show significant performance drops when ablated).

Thinking about how the model represents the location of the cat given the data from analyzing the queries, keys, values, outputs, and the head effects via path pactching, we can start to build a bigger picture of what is happening and start thinking about a circuit.

<br/>

```markdown
                [Early PTHs L2-L6] -----> [Mid PTHs L10-L12] 
                       |                        |
                       v                        v
                [DTH L8.1] -----------------> [IHs L14-L17]
                       |                        |
                       |                        v
                       |              [Late PTHs L16-L23]
                       |                        |
                       |                        v
                       +-------------------> [CSHs L14-L23]
```

```markdown
[Early PTHs L2-L6] <---> [Early PTHs L2-L6]  (Strong mutual interactions)
       |||                    
       vvv
[Mid Integration Hub L10-L12]
```

**Early Previous Token Head Processing (L2-6)**
- **Primary Function:** Initial semantic feature extraction
  - **QKVO Flow:**
    - 2.3 and 2.5's queries encode subject-object locations (`John cat`, `Mark cat`, `basket`, `box`) 
      - 2.3 keys add transition information to these encodings, reinforcing location (`John and cat`, `Mark and cat`, `cat and basket`, `room and John`, `room and Mark`)
        - 2.3 values project location/transition information forward to 2.5
    - 2.5 outputs encode subject-verb agreement with objects (`John takes cat`, `Mark puts cat`, `Mark leaves room`, `John comes back`)
    - 5.4 queries against 2.5's value patterns while integrating temporal context (`John away`, `when away`, `Mark leaves room`, `Mark goes work`)
      - 5.2 and 6.2 keys attend to more subject-action-location bindings (`John thinks`, `John takes`, `Mark takes`, `John room school`, `Mark room work`, `John leaves room`, `John puts cat`)
        - Values project refined semantic patterns forward

   
```markdown
[Mid PTHs L10-L12] <======> [DTH L8.1]
        |||                    ||
        vvv                    vv
[Induction L14-17]   [Copy Suppression Early]
```

**Mid-Layer Previous Token Integration (L10-12)**
- **Primary Function:** Complex state representation building
    - **QKVO Flow:**
      - 10.5 queries against 5.4 keys to encode basic but incomplete phrases (noun, verb, prepositional) with John-centric bias (`John takes the`, `Mark takes the cat`, `John comes back`, `John looks around the room`)
      - 10.5 queries 8.1 outputs, encodes parallel states between `John` and `Mark`
        - Keys draw from 8.1 output to track objects (`cat`, `box`) early and locations (`room`, `school`, `work`) mid-sequence
          
      - 11.3 queries the output of 2.3 to encode scene's initial state and individual subject perspectives
        - Keys and queries interact with 10.5 to attend to locations, objects, and scene states
          - Values project initial scene state from 10.5 queries (`the room there are John, Mark, a cat, a box, and a basket.`)
          
      - 12.1 queries track movement of main subjects via 5.2 keys with focus on John's state of mind after returning (`He doesn't know what happened`)
        - 12.1 keys heavily attend to the actions of the subjects before they leave and after they leave
          - Values concentrate on sequence during John's absence and his lack of knowledge after return
            
      - 12.2 queries the keys of 12.1, forms a strong previous token pattern across the entire sequence, most activity on (`John takes the cat and puts it on the basket`, `Mark takes the cat off the basket and puts it on the box`, `the cat is on the`)
        - 12.2 keys attend to the tokens in 12.1's queries forming the same pattern across the entire sequence, most activity in the same areas
          - Values encode mid-sequence events (`He leaves the room and goes to school`, `Mark leaves the room and goes to work`, `John comes back from school and enters the room`)
            
      - 12.2 queries the keys of 12.3, creates tight integration cluster across entire sequence, most activity on (`a cat`, `the cat`, `the basket`, `the box`, `the cat and puts it on the`, `the cat is on the`)
        - Values encode semantic state patterns
          - The final output of 12.3 balances attention between subjects, objects, and locations (`John takes cat and puts it on the basket`, `Mark takes the cat off the basket and`)


```markdown
[DTH L8.1] ---------> [Induction L14-17]
    |                        |
    |                        v
    +-----------------> [Copy Suppression early]
```

**Duplicate Token Head Processing (L8.1)**
- **Primary Function:** Parallel state perspective maintenance
    - **QKVO Flow:**
      - 8.1 forms a strong duplicate token pattern across the entire sequence
        - Queries the output of all previous token heads
          - Keys match against accumulated current and past location states
            - Values create a clear, dual, perspective-based representation from multiple inputs
      - Output maintains parallel current/believed states with heavy emphasis on both subjects


```markdown
[Induction L14-17] --------> [Late PTHs L16-L23]
        ||                          ||
        vv                          vv
[Copy Suppression Mid]  [Copy Suppression Late]
```

**Induction Head Processing (L14-17)**
- **Primary Function:** Temporal pattern recognition
    - **QKVO Flow:**
      - 14.2 queries against the values of 8.1's parallel states, focusing on initial scene state, Mark's cat-moving actions, with simultaneous emphasis on John's room inspection upon return
        - Keys attend to subject actions at key sequence points and targeting `John` moving the `cat`, post-moving actions and his return
        - Values emphasize John's full cat-moving actions while simultanously focusing on his return, and Mark’s final positioning of the cat
       
      - 15.0 queries the keys of 8.1's `box`/`basket` positions at initial position of the sequence, emphasizing `cat` movement and higher correlation with the basket
        - Keys match 8.1's queries with heavy emphasis on: `Mark` moving the `cat`, John’s actions pre- and post-moving, `John` searching for the cat
          - Values settle on `Mark` moving the `cat` and `Mark` leaving for `work`
      - 15.0 forms strong induction pattern
        - Queries keys of 12.2, focuses on all previous tokens, emphasizing subject's location changes (`work`/`school`)
          - 15.0 keys attend to 12.2 values, emphasizing subject's location changes
      - 15.0 queries 14.2 keys attending to `John` initially putting the `cat` on the `basket`, correlating with 15.0 simultaneously querying the inital state, each subjects perspective, emphasizing John and Mark's initial actions (`cat on basket`/`cat off basket`)
        - Keys attend to values, (`Mark leaves the room and goes to work. John comes back from school and enters the room`) high correlation to John's initial location change of the cat and Mark's actions
     
      - 17.6 queries 2.5's keys tracking `cat` position changes (`box`/`basket`) from both perspectives
        - Keys heavily attending to queries, captures action/temporal information across sequence, temporal markers highlighting what `John` doesn't see during his absence
          - Values capture keys of 2.5 and project `John`'s return to the `room` forward
      - Queries 8.1, 11.3, bringing a broad downstream update of refined semantics, and parallel subject processing
        - Keys attend to 8.1 token positions, massive emphasis on the initial state of the room
          - Values encode 8.1 keys equally across both perspectives
      - Sparse query signals from 11.3 emphasize `Mark` changing the `cat`’s location and focus on the `basket`, correlating with `John`’s return
        - Keys attending to queries returns a heavy emphasis from 11.3, simultaneously focusing on the `box` and `basket` with higher correlation on `basket`, and `John` coming back to the `room` and unaware of what happened
      - 17.6 forms a strong induction pattern across the entire sequence
        - Queries 15.0 keys for dual perspective encoding
          - Outputs refined semantics with high attention to dual perspectives.
         

```markdown
[Late PTHs L16-L23] <====> [Copy Suppression L14-L23]
            |||                    |||
            vvv                    vvv
         [Final Output]        [State Filtering]
```

**Late Previous Token Integration (L16-23)**
- **Primary Function:** Final state integration
   - **QKVO Flow:**
   - 16.7 queries the output of 2.3 isolating Mark's state and actions over the entire sequence
     - Keys attend to output and heavily attends to John's temporal state (`While John is away`, `doesn't know what happened while he was away`), actions, emphasizing the last phrase `John thinks the cat is on the`, there's signal of Mark's state when we moves the cat but the stregnth is very low. Keys also attend to queries and focuses on `basket` and `box`
       - Values project high strength on `basket`, `on`, `leaves`, `enters` from 2.3 outputs
         
   - 16.7 queries the keys of 10.5, focuses on determiners and adpositions, with a bias for the beginning of the sequence and the end
     - Keys attend to output and query to focus on auxiliary verbs and temporal markers with a John bias, simultaneously focuses on Mark leaving the room from output and John coming back to room from query
       -  Values project function words from the keys, with a bias on `John` and `Mark` initial state

  -16.7 queries the output of 16.2 emphasizing when John was away and when he came back, showing high correlation between each instance over the sequence
  - Queries the keys to encode where John may think the cat is, 16.7 attending to almost all the tokens in the last phrase of the sequence. Gives high attention to John's actions and possible cat locations
    -  Values encode where Mark put the cat, shows very little attention to John's perspective
 
  - 18.6 queries from all processed streams, primarily from induction head outputs
    - 5.4's keys focus on Mark moving the cat with lots of strength on the cats new location, simultaneously attributing lots of attention to what John missed while he was away, coming back and the final phrase of the sequence
    - 18.6 keys encodes 5.4's queries, where the queries encode John's inital state at the very beginning, when he originally places the cat, when mark moves the cat and when John comes back and the keys attend to this
    - Values project this information forward
    - 18.6 queries the keys of 15.0, encoding the shared relationship between sequence positions highlighting John leaving the room and going to school and Mark leaving the room and going to work, especially focusing on when Mark left and when John put the cat on the basket
    - Values projecting queries about John and Mark coming and going, and the cats position throughout the sequence
      
    - 22.2 queries 2.5 output, encodes shared relationship between positions marking John and Mark's departure, John's return, with a focus on phrases like `He leaves the room`, `what happened in the room when he was away`, `John know`, `doesn't know what happened`
    - Keys attend to queries focusing on the initial state of the room, temporal markers like `While`, and `what happened in the room while he was away`
    - Values to queries encode `in` and `box` heavily, values to keys encodes `John` sparsely, `Mark` densely, `cat`, and `comes`
    - 22.2 queries 16.2 keys, focusing on the initial state, John originally placing the cat, Mark moving the cat and the final phrase of the sequence (`the cat is on the`) with emphasis on temporal markers where most attention are on Mark's actions
    - 6.2 queries encode John's unawareness of what happened while he was away, and 22.2 keys attend to pronouns and cat locations
    - 18.7's keys attend to the initial state of things, John and Mark moving the cat and the final phrase of the sequence, 22.2's queries show strong emphasis on where Mark moved the cat
    - 18.7's queries encode when John returns, while 22.2's keys strongly attend to Mark's actions in moving the cat and his departure from the room
    - 22.2 values project the output of 18.7 in regards to Mark's actions beyond the initial parts of the sequence forward

    - 22.2 queries the output of 21.5, focusing on the initial state of John, Mark and the cat, positions in the sequence where the cat was moved, both with strong attention on John's unawareness and the `box`/`basket`'s relation to the final phrase of the sequence
    - Queries focus on keys which attend primarily to `John`
    - 22.2 keys attend to the output with a heavy focus on the cat being on/off the basket, the cat being on the box, and most of the attention on John's unawareness
    - Keys attend to the queries with a primary focus on the final phrase in the sequence and the initial state John, Mark, the cat and the room
    - Values attend to the output and focus on the on/off relationship to the cat and objects

    - 22.4 queries encode 2.3 keys that attend to all cat locations in the sequence, physical location of the subjects, the subjects departures and the unawareness of John
    - Keys attend to queries, focusing on John's initially placement of the cat on the basket, Mark moving the cat, with a strong focus on the keys attending to Mark moving the cat and the queries encoding John's unawareness
    - Values encoding and projecting the the output where its focused on John in the beginning of the sequence, his actions throughout and heavy attention on John thinking
   
    - 22.4 queries the keys of 8.1, where the keys heavily attend to the initial mention of `John` and all subsequent mentions with varying attention strength. The queries encode the initial phrase of the sequence
    - 22.4 keys attend to 8.1 queries, focusing on duplicate/similar phrases, with high attention on temporal markers
   
    - 22.4 queries the keys of 10.5, where keys heavily attend to `Mark takes the cat off the basket and puts it on the box` and `John looks around the room`, while the queries focus on John putting the cat on the basket and leaving the room and John's state at the beginning of the sequence
    - Keys attend to the output, where most of the focus is on John moving the cat, and John's unawareness, which is heavily attending to `John takes the cat and puts it on the basket`
    - Values project this information as well as temporal markers forward

    - 22.4 queries the output of 15.0, both focusing on the seperate perspective of Mark/John and their actions toward the cat
    - 22.4 queries the keys of 15.0, both focus on the state of the room at all positions in the sequence, with a heavy focus on John's unawareness of the room while he was away
    - 22.4 keys attend to 15.0 outputs, focusing on John/Mark leaving the room, with most attention on Mark's action, John unawareness and John's initial actions
    - Keys attend to queries, focusing on the cat at the beginning of the sequence, and from John/Mark's perspective
   
    - 22.4 queries, focusing on John/Mark moving the cat between the basket/box, query the output of 16.7, which focuses on John leaving for school, with more focus on John returning
    - Queries encode the keys, focusing on determiners and their relationship to subjects, objects, locations, with dense strength on actions taking place in the room
   
    - 22.4 queries the output of 18.6, focusing on entire phrases. The output, where John initially places the cat, highly correlating with his unawareness of how things changed denoted by the queries, Mark moving the cat and Mark leaving the room
    - Keys attending to queries and highly correlating between Mark leaving the room and the initial state of the room
   
    - 22.4 queries the output of 18.7, output focuses on John being away, coming back, and where he thinks the cat is, queries focus on the initial state of the room, and John/Mark moving the cat, their actions and John unawareness correlating to the last phrase of the sequence
    - Queries encode the keys, heavily attending to John/Mark and the initial state of the sequence
    - Keys attend to queries, heavily focused on the initial state of the room in relation to John looking around the room and his unawareness
    - Values heavily encode Mark's actions
   
    - 22.4 queries the output of 20.2 and focuses on Mark's actions in relation to John's
    - Keys heavily attend to the `leaves` positions in 20.2's values. The keys also attend to Mark's action of moving the cat in 20.2's output, which is correlating with John leaving the room and returning from school
    - Values are sparse and encode the initial position of the basket in 20.2's output, query, while encoding temporal information from the keys

    - 22.4 queries the output of 21.5, focusing on the state of the entire sequence as John is away at school, with most attention on the initial state of the room at the beginning of the sequence
    - Keys attend to the output, where the initial state of the room is correlating highly with John's actions, ignoring most of Mark's actions. Keys also attend to queries focusing on John leaving, comparing that to every phrase in the sequence with the most focus on John's return and unawareness of changes made by Mark
    - Values draw from the output and focus heavily on the initial state of the room
   
    - 22.4 queries the keys of 22.3, focusing on the initial state of the room and John leaving the room with strong attention across John/Mark's perspective when they seperately moved the cat
    - Keys attend to the output, showing a more focused representation of the Q/K relationship. Keys also attend to queries, showing the same relationship, but with heavy attention on Mark leaving, John returning and how those perspectives correlate to John's unawareness of the cat's new position

    - 22.5 queries keys of 5.4, focusing on `John comes back`, `knows` and `the cat`, while simultaneously attending to John/Mark leaving the room
    - Keys attending to queries, focusing on the initial state of the room and John initially moving the cat
    - Values are sparse but encode the keys and projects John coming back to the room and `school` with heavy attention
   
    - 22.5 queries the output of 6.2, encodes token positions related to `John`, `the room`, his actions and temporal markers. Queries then focus on keys and encode token positions related to the initial state of the room, John being away, Mark moving the cat then leaving and John returning
    - Keys only attend to queries related to John/Mark leaving, and John return
    - Values only project temporal markers from the keys forward

    - 22.5 queries the keys of 14.3, keys attend to Mark leaving the room, queries encode that, John's unawareness and heavily focuses on instances of `the basket`, `the box`, `the room`, `the cat` across the entire sequence
    - Keys attend to output token positions, correlating John initially moving the cat, with most of the attention on Mark moving the cat and leaving the room

    - 22.5 queries the output of 16.2, focusing on John and Mark moving the cat, and the initial state of the room, with most of the attention on John. 16.2's keys attend to John's unawareness while 22.5's queries heavily focus on Mark moving the cat

    - 22.5 queries the output of 18.7 showing equal strength to the initial state of the box/basket while encoding the position that marks John's unawareness
    - Keys heavily attend to outputs regarding John's unawareness and his action of looking around the room. Keys also attend to the inital state of the room from the query positions
    - Values receive 18.7's output and projects heavily activated `Mark` tokens

    - 22.5 queries the keys of 22.3, sparsely focusing on `cat` across the sequence
    - Keys attend to the output, focusing on John/Mark moving the cat. Keys also attend to the query, showing heavy correlation between John/Mark leaving and John's unawareness
    - Values encodes prior queries forward 
       
     - 21.5, 22.2/3/4/5, L23.H6 perform final integration by:
        - Querying against induction, copy suppression and duplicate token head outputs, primarily from 15.0, 16.7, 17.6, 18.6, 18.7, 20.2
        - Keys matching belief states
        - Values producing final predictions

**Copy Suppression Processing (L14-23)**
- **Primary Function:** State filtering and arbitration
    - **QKVO Flow:**
      - 14.3 queries check current states against induction patterns for initial state filtering
      - 16.2 and 18.7 perform intermediate filtering
      - 20.2 handles mid-circuit filtering
      - 23.5 performs final arbitration through:
        - Queries verifying belief states (`John cat on basket`)
        - Keys checking reality states (`Mark cat on box`)
        - Values suppressing inconsistent predictions
        -  - 8.1 Output maintains parallel current/believed states. When 16.2's keys interact with 8.1's output, activations correspond to (`John`, `Mark`, `cat`, `box`, `basket`) in the beginning of the sequence. Disperse activations for temporal and action tokens
          - Activations correspond mostly to `Mark`
        - 16.2 receives 8.1's output, duplicate token head informs suppression head of duplicate activity. 16.2's output suppresses repeated names as suppression activations for Mark's repeated tokens are higher than John's, mitigating the actual state of the cats location in favor of the belived state with clear separation 

A more in-depth analysis of the QKVO-dimensions can be found <a href="https://github.com/christianThardy/christianThardy.github.io/blob/master/tom-circuit-path.md" title="ToM circuit paths" rel="nofollow">here</a>. Thinking about the circuit from a high level:

**Previous Token → Duplicate Token:**
The outputs from early previous token heads are fed as queries, keys, and values into the duplicate token head (8.1). By capturing the same tokens from multiple angles, 8.1 maintains parallel, multi-perspective state representations—one for each subject or belief context—enabling the model to track what each subject knows or *believes*.

**Duplicate Token → Induction:**
The multi-perspective states generated by 8.1 are then passed to the induction heads (L14–17). Here, they serve as queries that tap into specialized key-value patterns, refining each subject’s actions, beliefs, and locations. This “induction” step crystallizes temporal relationships (who did what, and when) into coherent narrative arcs.

**Induction → Copy Suppression:**
Once the induction heads have established these evolved belief states, they flow into the copy suppression heads (L14–23). The suppression layers use *belief*-oriented queries and keys to filter out redundancies or conflicting states (e.g., repeated mentions of Mark’s actions vs. John’s). This ensures the final narrative tracks each subject’s perspective accurately without over-amplifying duplicates.

**Copy Suppression → Final Integration:**
The output from the suppression phase is handed off to the Late previous token heads (L16–23) for the ultimate integration. These heads arbitrate among the refined beliefs, subject-object bindings, and temporal events, consolidating them into a single coherent representation. The value vectors at this stage crystallize the final model output, ensuring that both real and believed states converge into an internally consistent conclusion.

The full circuit evolves from early semantic feature representations into layered belief-action integration, duplicate token heads maintain parallel representations of reality, induction heads refine these parallel states over time capturing cause-effect or before-after relationships, copy suppression ensures the model doesn't mistakenly merge conflicting beliefs and the final previous token heads produce the final prediction, melding all subject-object-location states into a unified narrative.

Each layer builds on prior patterns, maintaining Mark’s actions as current-world events while keeping John’s beliefs separate. The circuit appears to maintain a fundamental asymmetry between the two subjects—highlighting a meaningful cognitive distinction. The negative effects from ablation studies (particularly around 8.1 and 12.3) reveal critical integration points where parallel processing streams must be correctly combined to maintain accurate belief tracking.

The system balances belief preservation and action-driven updates, forming a dual-representation architecture, tracking what Mark does to know the true state, what John believes to make the final prediction, and maintain the separation between these two representations—ultimately yielding a model output that can differentiate between actual events and each subject’s belief or knowledge state.

<br>

### Dictionary learning, sparse autoencoders and superposition
<sub>[↑](#top)</sub>

The linear representation hypothesis tells us that activations are **sparse**, **linear** combinations of **meaningful feature vectors**.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/102697598-f4d78700-4204-11eb-837c-40fb733f52df.gif" width="550px">
</p>

<br/>

Dictionary learning aligns closely with the linear representation hypothesis<sub>[<a href="https://transformer-circuits.pub/2023/monosemantic-features/index.html" title="Bricken" rel="nofollow">18</a>]</sub>, aiming to express complex data as a linear combination of simpler elements, or "basis vectors". These basis vectors form a dictionary—a data structure that holds key-value pairs—and when combined can efficiently represent the original data, making it easier to analyze, compress, or reconstruct. In models, a dictionary of learned concepts with associated directions allows specific elements to be activated based on relevance to the input; for example, `queen` could be represented by a combination of `female` and `royalty` directions. Sparsity is key here, as most concepts are irrelevant to a given input, resulting in many feature scores remaining zero.

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

So we can take the activation vectors from attention, an MLP or the residual stream, expand them in a wider space using the SAE where each dimension is a new feature and the wider space will be sparse, which allows us to reconstruct the original activation vector from the wider sparse space, then we get complex features that the mechanism has learned from the input<sub>[<a href="https://transformer-circuits.pub/2024/scaling-monosemanticity/" title="Templeton" rel="nofollow">19</a>]</sub>. From this we can extract rich structures and representations that the model has learned.

The SAE suite used is Google Deepmind's <a href="https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/" title="Google Deepmind" rel="nofollow">Gemma Scope</a>, and the output was visualized using <a href="https://docs.neuronpedia.org/" title="Neuronpedia" rel="nofollow">Neuronpedia</a>. Gemma Scope is a collection of hundreds of SAEs on every layer and sublayer of Gemma-2-2B and 9B. Using the trained SAE on the ToM passage, we can take features from layer 22 of Gemma-2-2B out of superposition, and see which features in the model are activated.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/2c554f22-7de0-4b2b-9f5e-2a30faef77b3" width="480"/>
</p>

<br>

The model has specific features dedicated to representing different aspects of the narrative in the residual stream. For example, feature 61 focuses on *references to positions and locations in a narrative*. This feature has a high explanation score<sub>[<a href="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain" title="Bills" rel="nofollow">20</a>]</sub>, showing that the model is correctly isolating different narrative elements through distinct features.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/285680ab-c15e-46f6-9c3d-f963430fe969" width="480"/>
<img src="https://github.com/user-attachments/assets/73540c29-3935-4b85-aeff-7a2b65a738f7" width="480"/>
</p>

<br>

These features suggest that the model is building an internal representation of the physical setup described in the passage, tracking where objects and subjects are placed. It’s also clear that several features are responsible for keeping track of John and Mark's movements and actions.

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

Several features seem to be directly tied to representing belief states and knowledge. Feature 13597 is likely crucial for capturing John's lack of knowledge about what happened in the room while he was away. Feature 5107 probably signals the model’s awareness of John’s ignorance, potentially reflecting uncertainty and doubt. Feature 12703 could be involved in modeling John’s thought process when he returns to the room, helping the model represent how John updates his beliefs. These features seem important for understanding how the model processes ToM scenarios, especially when tracking subjects’ evolving mental states.

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

As a rough analogue to how neural fMRI scans capture distributed activations, attention heads shift focus across tokens, similar to how brain regions activate based on focus and task demands. We can make this analogy by thinking about the parallels between functional lobes in the brain and the structure of a transformers attention mechanisms. Each brain lobe has a specialized role: the occipital lobe handles vision, and the frontal lobe manages planning. Attention heads work similarly, processing contextual knowledge within specific structures. Like lobes aiding decision-making by accessing relevant knowledge, attention heads enable transformers to weigh parts of the input sequence. 

If we zoom out from any single head, we can define specific attention heads across layers as circuit components. From there, we can start mapping out how these components *fire* across the ToM passage, revealing how they work together to solve the task. The methodology aligns closely with the original paper, but with some tweaks: activation data is collected, co-occurrence metrics are calculated, spectral clustering is applied, and affinity matrices with the Phi coefficient are used with spectral clustering. Tests were run on a small dataset that uses different templates to construct false belief passages that structurally resemble the original ToM narrative.

The results show distinct ToM subcircuits—sets of attention heads lighting up at key points during the task. These components act as cohesive units, each one relative to others, activating or staying dormant at different sequence positions. This makes it possible to see which components have groups of heads that activate together across different contexts, and allows us to see how information flows through the network as its making its predictions. For example, within the action-location state, certain heads may consistently activate with suppression heads, particularly when managing changes in the scene and beliefs about the scene in the penultimate state. By calculating these affinities, its possible to see which specific heads within each component interact most frequently, giving insight into sub-patterns within the larger components. Think of it like t-SNE but for attention circuits.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/86f79bc4-1a5b-4f27-bee2-95fea1f616d3" width="800"/>
<br>   
<small style="font-size: 12px;">High activation values indicate components that are more activated against low activation values.</small>
</p>

<br>

Starting with the co-occurrence matrix, all of the heads selected from the causal analysis are firing together on a sliding scale, suggesting they work together to maintain and update subject state information throughout the sequence. The action-location state (duplicate token heads) shows particularly strong negative co-occurrence with late suppression heads, indicating these components work together to have opposing functions—one tracking the actual state while the other partially suppresses information contradictory to John's perspective. 

While late suppression heads have strong negative co-occurrence with action-location state heads, they have relatively neutral co-occurrence with initial subject state heads, suggesting that late suppression isn't blanketly suppressing all prior information, but rather selectively targeting action-location information—perhaps helping to maintain the “false belief” by specifically suppressing the true location information while preserving the subject's initial state understanding.

There's a clear hierarchical structure in the suppression heads (early, mid, and late), with moderate co-activation between early and mid suppression heads, but distinct activation patterns for late suppression heads. This highlights early suppression during initial state filtering, mid suppresion during pattern refinement by the induction heads and late suppression applying the final arbitration before prediction.

We can see this mimicked in the temporal activation plots but with more detail. Cluster 2 (action-location state) shows the strongest activation during the intermediate state, suggesting this is when the model is most actively processing the shared semantics from the previous token heads. This makes sense as this would be when the model needs to track the actual movement or position changes in parallel between both subjects, where it learns the initial location of the cat, which is important from subject 1's perspective. The suppression heads (Clusters 0, 4, and 5) show interesting temporal patterns where late suppression heads are most active in the penultimate state, early suppression heads show the most activation in the initial state, and mid suppression heads maintain relatively consistent activation levels, but shows the most activation in the final state. Perhaps suggesting the patterns learned in the middle layers were important to later layers, but need to be suppressed right before prediction.

The intermediate subject state (induction heads, Cluster 6) shows selective activation during the initial and final states, suggesting they may be crucial for maintaining and updating the model's representation of the subject's knowledge that it learned during the beginning of the passage, when the model needs to recall and apply patterns from earlier in the sequence to recall the initial state to predict John's belief. Information is then transferred to the final state to co-activate with the previous/duplicate token heads and suppression to predict the correct location. So it's important for connecting earlier events back to later states.

The final subject state heads (Cluster 1) show subtle, increasing activation from the intermediate to final state, suggesting they integrate information to form the final representation of the subject's belief states. Also, the intermediate subject state (induction heads) shows an interesting dip in activation during the penultimate state, right when the early suppression heads (Cluster 4) show increased activation. This could indicate a mechanism where the model temporarily suppresses ongoing state tracking right before the final state computation—perhaps a kind of "reset" before integrating all the information for the final belief state determination.

Looking at the distribution of activation strengths across clusters in the second temporal plot, it's noteworthy that the action-location state shows the most extreme activation values. This suggests that tracking the parallel states of each subject's physical state/location in relation to the believed location of the cat, and integrating those semantic features in context might serve as a kind of “ground truth” against which subject states need to be compared.

Everything here aligns with the QKV patterns seen from the ablation studies and interventions. The temporal activation patterns provide additional evidence that previous token heads serve as foundational sequential processors, and induction heads act more like specialized pattern recognition and recall mechanisms that are particularly important for handling long-range dependencies in the false belief task.

This again is confirming that the model maintains multiple representations of reality (actual locations) and beliefs (subject states) through coordinated activation of different head clusters.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/0636d720-c9af-4928-a232-0ce58b2c6e9a" width="650"/>
<br>
<small style="font-size: 12px;">Theory of Mind Circuit</small>
</p>

<br>

The ToM circuit efficiently balances between retaining initial knowledge, updating as the story progresses, and suppressing outdated information. This aligns with human-like belief updating, where new observations modify existing beliefs without completely discarding past knowledge. It’s especially crucial for false belief tasks, as it supports reasoning about beliefs that differ from reality—understanding what John believes (`cat on basket`) versus what is actually true (`cat on box`).

The full circuit reveals a nuanced algorithm in its attention:

- **initial subj state (previous token heads)** identify early occurrences of the same tokens that immediately precede the current one that represent locations, subject actions, objects and positions in relation to John and Mark.
    - e.g., cat in room, box in room, basket in room, John in room, Mark in room
      
- **action-location state (duplicate token heads)** captures all prior local dependencies, primarily focusing on locations of subjects and objects with equal weight, to repeated tokens, placing them in the context of the ongoing scene and keep their states parallel.
    - e.g., John puts cat on basket then leaves room, Mark puts cat on box then leaves room, John returns to room, John goes to school, Mark goes to work
      
- **intermediate subj state (induction heads)** captures long range dependencies from duplicate token output, maintains the state of subjects' in the scene by detecting patterns, copying and propagating tokens forward from early tokens previous positions in the sequence.
    - e.g., John put cat on basket, John at school, John not in room, Mark not in room, John back from school and enters the room, cat currently on basket
      
- **early, mid and late supression heads** negatively effects true-beliefs and prevents copying the actual location of the object via negative modulations from value vectors at different points in the sequence.
    - e.g., John put cat on basket, John at school, Mark takes cat off basket, Mark put cat on box, John not in room, Mark at school, cat currently on box (according to Mark's belief), cat currently on basket (according to John's belief)
        - John+++, Mark+, cat on basket++++, cat on box--
     
- **final subj state (previous token heads)** query against induction and copy suppression outputs and produce the final prediction.
    - e.g., John thinks the cat is on the basket

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/6a29ec72-9f80-49c6-9495-4be646fc0c73" width="650"/>
</p>

<br>

The early layers, or initial subject heads, mostly handle simple linguistic elements (parts-of-speech, puncuation, determiners, conjugations, function words, syntactic dependencies) in specialized later heads.  These heads focus on picking up broader contextual signals, with key vectors usually having a larger influence. This suggests that early layers are primarily focused on gathering broad, diffuse information and maintaining generalized semantic information.

As we move into the middle layers, things get more interesting. Here, the action-location state heads start doing more compositional work, integrating outputs from subj-1's state and subj-2's state. This is where object tracking, action understanding, and structural processing begin to take over. The attention mechanism becomes more balanced between the query and key vectors, indicating a shift towards integrating contextual information more precisely and building up a richer understanding of the parallel perspectives of the scene.

This scene understanding flows into induction heads, especially for entities like John and Mark, where the model begins to track complex subject-object interactions and manage belief states—continuing to maintain the broader context built up by their initial head states, and the action-location state. It’s here that we see the emergence of complex reasoning and specialized attention heads, such as tracking belief states while keeping attention on earlier elements of the narrative in relation to John.

At the final stages, the suppression heads play a key role. They show both positive and negative modulations between the QK mechanisms, enhancing and inhibiting specific connections as needed. Here, the value mechanism filters out outdated or irrelevant information to John's knowledge, ensuring only relevant factors—like John’s incorrect belief about an object’s location—are propagated to influence the model’s final output.

So the model builds the subject's false belief about an object’s location by: **1)** Identifying John as the belief holder. **2)** Tracking the cat's movement. **3)** Updating its knowlege on object locations. **4)** Integrating these elements into John's belief state. **5)** Suppressing information irrelevant to the belief holder.

The ToM circuit satisfies the three criteria discussed in Wang et al<sub>[<a href="https://arxiv.org/pdf/2211.00593" title="Wang" rel="nofollow">10</a>]</sub> . Minimality demonstrates each head’s contribution to ToM capability via its direct impact on logit differences by component. The score, reflecting the percentage of the model’s total logit difference (0.8365) attributed to each head, highlights the importance of each head to the task.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/87a11ac9-959d-45e8-9bac-cb0cb1b546f2" width="700"/>
<br>
</p>

```markdown
Average logit difference (ToM dataset, using entire model): 0.8365
Average logit difference (ToM dataset, only using circuit): 0.9373
```

<br>

The ToM circuit hits all the key benchmarks: faithful—the circuit actually outperforms the full model, showing it captures the necessary functions; complete—all heads essential for each component are included; minimal—the plot highlights clear specialization with only a minimal number of heads carrying substantial weight.

Breaking it down, the ToM circuit shows concentrated importance in certain heads, with over 40% in the previous token heads. This suggests that these heads are keeping a coherent grasp of where John thinks the cat is, and is critical for handling ToM tasks.

Meanwhile, the duplicate token heads contribute minimally, acting more as a supporting context provider rather than the main drivers of belief tracking.

The circuit also shows a high degree of modularity: heads are highly specialized, with relevant computations neatly contained within each component. This limits interdependence with other network parts outside the defined circuit, showing a clean and compartmentalized structure.

<br>

#### Copy supressions role in the ToM circuit

Copy supression[<a href="https://arxiv.org/pdf/2310.04625" title="McDougall" rel="nofollow">22</a>] in the ToM circuit are heads in the model that respond to predictions made by prior heads and adjust the final output prediction negatively. These heads have the advantage of seeing all preceding context and intermediate predictions generated so far. By leveraging this, they can calibrate the model's confidence in predicting the next token, effectively fine-tuning the logits before the final prediction is made.

Copy surpression in later layers operates in the unembedding space of the model. Consider an induction head that's tracking the belief state. Suppose the model processes the sentence: `John put the cat on the basket`, and the current token is `the`. The induction head predicts "basket" as the next token based on the context. This prediction is written to the residual stream and will be mapped to the logits for the final output. However, before the model commits to this prediction, the copy suppression mechanism kicks in. It performs post-processing on the logits by suppressing any outputs that have been previously seen but aren't relevant to the current context established by the induction head. 

Essentially, while some heads focus on specific tasks—like predicting the next word based on the context of previous next word predictors—other heads monitor the earlier predictions and adjust them, ensuring the model doesn't over-rely on copying tokens that aren't contextually appropriate. The degree of copy suppression is influenced by how much attention the model pays to the tokens it's considering copying. This aligns with the iterative nature of LLMs. They refine their predictions layer by layer, with each layer building upon the representations from the previous ones as information flows toward the final layers. 

This is purely speculative, but I suspect the model might have the capability to represent second-order false beliefs—essentially, understanding that one person can hold a false belief about another person’s belief. This could emerge from its ability to juggle parallel representations of both true and false information, potentially through mechanisms like copy suppression.

There's a lot more we do not know about these heads and they probably have more complex circuitry that describes when it is good to surpress information and when it is bad. 

<br>

### Ablation studies <a id="ablation-studies"></a>
<sub>[↑](#top)</sub>

Ablation studies are widely used in neuroscience and they are super useful for neural networks as well. The idea is to systematically “remove” (or ablate) specific mechanisms—like neurons, layers, or attention heads—within the model to assess their contribution and see how much they really matter to overall performance. 

When we mean-ablate the entire ToM circuit, performance drops by about 80.66%, showing a massive reduction in the believed-actual difference of the model's inference accuracy—the model's confidence of the `basket` token as the correct prediction.

```markdown
Full Circuit Mean Ablation Results:
Number of heads ablated: 28
Original believed-actual diff: 0.836511
Ablated believed-actual diff: 0.162061
Total circuit effect: 0.674451
```

This suggests that these heads are working together in a highly interdependent way. The remaining performance (~16.20%) implies that outside the ToM circuit, there’s not much capacity left for correct prediction of ToM tasks, as expected. Unsurprisingly, John's duplicate token belief state heads and the early suppression heads come out as the most critical. Ablating these reduces performance by ~11.01% and ~45.88% respectively.

<br>

## So What?
<sub>[↑](#top)</sub>

There are key interactions and patterns that we can see backed by qualitative evidence. 

Circuit components have complementary timing in the way they activate across the sequence. The action-location state activates strongly in middle and later layers, components complement each other during belief processing, subject states and inhibition head clusters show complementary patterns; they tracks beliefs, and the other tracks what's not believed. Components are processed sequentially. Previous token heads provide steady baseline processing, induction heads build up activations over the sequence, and copy suppression prevents simple copying at the end.

Out of 175 total attention heads in Gemma-2-2B's attention mechanism, there are 28 that display a significant increase in ToM performance when isolated, and a significant decrease in model performance when they are ablated. This is not an isolated result. In a separate study, element-wise analysis of LLM neurons have been found to show increased firing rates for isolated sets of neurons when performing ToM tasks when compared to isolated human neurons that show consistent fire rates across similar false-belief tasks<sub>[<a href="https://arxiv.org/pdf/2309.01660" title="Jamali" rel="nofollow">23</a>]</sub>. In both cases showing modularity and a parallel to the human brain.

From the perspective of this task, copy suppression helps the model maintain separate representations between what is actually true (reality) and what is believed to be true (beliefs), and this could have several implications for AI alignment. Because the model has learned to maintain distinct representations and track multiple potentially conflicting “versions of reality” this highlights the capability for nuanced reasoning—understanding different perspectives, and possibly even lying. Investigating inhibition and suppression mechanisms might be crucial for understanding how models might deceive, but these same capabilities could be incredibly useful for alignment research. For example, they could help with:

- Value learning: Separating “is” from “ought” to reason about values.
- Goal preservation: Keeping different types of goals or beliefs separate and coherent.
- Corrigibility: Distinguishing human beliefs from reality, and recognizing the gap between “what is” and “what should be”. 

Could copy suppression be useful to improve alignment techniques and safeguard against belief corruption? This raises more key questions: how reliable is this mechanism for alignment? Can it scale to more complex belief systems? What are the failure modes, especially in edge cases?

Each component serves a specific role at different points in the sequence. The timing and strength of the activations suggest a well organized circuit that tracks states, actions, beliefs using linguistic elements throughout the narrative.

<br>

# Broader implications <a id="broader-implications"></a>
<sub>[↑](#top)</sub>

A common critique of LLMs is that they rely purely on formal linguistic competence, and therefore can't truly "learn" meaning in a deep sense<sub>[<a href="https://aclanthology.org/2020.acl-main.463.pdf" title="Bender" rel="nofollow">24</a>]</sub>. However, when considering *emergent understanding*—the idea that models develop an implicit sense of meaning based on patterns in the data—It begs to question: How do mechanisms effectively capture semantics to succeed at ToM? 

One plausible hypothesis is that while induction heads primarily track formal patterns, semantic meaning embedded in those patterns gets absorbed through training. For example, repeated references to “the cat being on the basket” provide a robust contextual anchor. Although induction heads focus on sequence-level correlations, they often align with real-world semantics present in the training data. When a model predicts that “the cat is in the basket”, it might be leveraging a weakly implicit form of semantic understanding (functional competency) encoded in its layers.

This idea is particularly relevant in tasks requiring predictions about mental states or perspectives. Even if the model initially exploits high-level patterns, these patterns often align with semantic reasoning. For example, deeper layers—say, layer 22—don’t just pass through raw pattern data from earlier layers. Instead, they integrate signals representing a mix of formal linguistic structure and contextual cues. By this stage, the model might be blending formal reasoning with the semantic relationships encoded in the data.

This raises another question: When the model predicts John’s perspective in a ToM task, is it actually reasoning about John’s mental state (functional competence)? Or is it just leveraging high-level linguistic correlations (formal competence) that happen to align with correct answers? I think there’s a blurry line here—meaning can emerge from form when structured, implicit grounding exists in the data.

Induction heads, while not explicitly designed to handle grounded semantics, may approximate grounding by exploiting consistent statistical patterns present in the training data. For example, if the model frequently encounters phrases like “John thinks the cat is in the basket” followed by predictable narrative outcomes, it could learn to associate these patterns with semantic relationships. By layer 22, earlier layers have already processed and encoded contextual cues such as entity roles and spatial relationships, enabling deeper layers to recombine these representations into contextually appropriate predictions. This process reflects how large language models can appear to reason about meaning despite lacking explicit semantic grounding.

Even without explicit grounding, models trained on structured datasets can still encode weak semantic signals. Benchmarks like MMLU, ARC-C or Winogrande embed linguistic patterns that implicitly carry semantic entailments or logical structures. Models like Gemma-2-2B seem to capture these relationships effectively, even if they’re operating formally. By layer 22, relational data synthesized from earlier layers yield outputs that mimic semantic understanding.

Tasks like Winogrande make this particularly clear: While solving these tasks seems to require semantic reasoning, models often succeed by exploiting subtle textual cues embedded in the data. This suggests that while the induction heads found in this analysis might not directly access labeled semantic relationships, they capitalize on implicit signals encoded in the training data. For example, co-occurrences of specific token patterns might encode semantic entailments without the model ever “knowing” what those entailments mean explicitly.

In large models like Gemma-2-2B, emergent semantic inference seems plausible due to the interplay between the architecture and the training data. Benchmarks like BoolQ and TriviaQA provide structured patterns that tie linguistic forms to functional outputs, creating a complex statistical scaffolding that weakly approximates grounded understanding. While induction heads and specific layers remain pattern-driven, the broader training process imbues the model with enough implicit grounding to perform tasks requiring nuanced semantic judgments. This bridges the gap between form and meaning, allowing the model to encode partial grounding—even if it never reaches full semantic understanding.

<br>

# Conclusion <a id="conclusion"></a>
<sub>[↑](#top)</sub>

By bridging high-level behavioral analogues (tracking and updating belief states of entities) with low-level computational mechanisms (transformer attention heads, MLPs and residual streams), the hope of my work here and future work is to validate or invalidate that certain heads or circuits are causally implicated in tasks that map onto ToM-like reasoning.

The proposed ToM circuit:

- Extends on the <a href="https://www.alignmentforum.org/posts/3ecs6duLmTfyra3Gp/some-lessons-learned-from-studying-indirect-object" title="alignmentforums" rel="nofollow">IOI</a> (focuses on tracking a models ability to reconstruct the syntax of natural language) work to identify specific attention heads that are pivotal to false belief tasks. The proposed circuit tracks and updates belief states of entities in regards to locations and objects using strong formal linguistic competence and tentative functional competence via the manipulation of linguistic elements, to distinguish facts from the believed reality of a 3rd person perspective.
  
    - The empirical results from activation/path patching identified a circuit that’s causally linked to ToM performance, and provides *some* causal evidence that form can carry function and certain heads are necessary for successful ToM-like inference. The circuit captures stable relationships (like who believes what) that go beyond surface-level token transitions. *Some* emergent semantic-like behavior exists—removing specific heads consistently reduces performance—which pushes beyond correlation towards a stronger (though still not definitive) causal story.
      
    - The fact that patching in certain Q, K, or V components from a “clean” run restores correct predictions indicates these attention heads are doing a bit more than just memorizing surface patterns. The heads appear to encode aspects of *perspective* and *belief* in context. The strong improvements following targeted interventions suggest the model internally represents subtle cues needed for ToM tasks. This finding is still a step short of indisputable evidence for genuine semantics, but beyond naive statistical correlation.

- Is robust to targeted ablations. Critical heads responsible for ToM capabilities were isolated to validate the circuit, and the observed performance degradations and full task recovery following ablations affirm the importance of these components in maintaining robust ToM functionality.
  
    - Furthermore, experiments show that when certain tokens (those involved in representing “belief states” like where John thinks the cat is) are patched from a clean run, the corrupted model’s performance on the ToM task recovers.
    
    - The removal of duplicate token and induction heads severely impairs ToM performance. These heads ensure that “belief tokens” and “action-location tokens” are managed distinctly, preventing confusion between real states of the world and an agent’s belief.
    
    - Demonstrates a direct causal relationship between linguistic representations and ToM task performance.

- Works with copy suppression to ensure that distinct belief representations are tracked and preserved, preventing conflation between reality and differing subjects' beliefs. The circuit's interplay allows for more accurate predictions of behavior based on mismatched beliefs, a hallmark of human ToM.
      
    -  This has several implications, one being that the model's internal representations are doing more than just predicting the next token. As the model processes the sequence, it maintains a belief about the entire future, not just the next word. If the model reads a sentence like *“John hid the cat in the basket, but Mark moved it to the box when John wasn’t looking”*, it has to keep track of where the cat might be (belief state) to predict any future reference to the cat’s location, even several sentences later.
      
       - Copy suppression allows the transformer to model this process explicitly by preventing overcommitment to any one interpretation of reality. This enables the model to maintain belief dynamics similar to how humans mentally track both reality and agents' beliefs about reality.

- Weakly shows that as LLMs scale and learn dense correlations, they develop weak semantic grounding—patterns that mimic *genuine* semantic and pragmatic reasoning.
  
    - The ToM circuit appear to track particular tokens (like “basket” vs. “box”) consistently, carrying forward these representations across layers and contributing to final predictions. This suggests the model is doing more than superficial form matching; it’s maintaining stable semantic relations that resemble an understanding of the narrative.
      
    - While speculative, experiments show that removing certain heads disrupts semantic coherence. Similar to visual psychophysics, where knocking out features tests perceptual encoding, altering heads in transformers reveals how ToM directions encode context. While it’s not definitive proof that the model truly *understands* semantics, it’s a concrete demonstration that complex formal pattern capturing is sufficient to manifest in behaviors associated with semantic interpretation.

The parallels to human thinking are fascinating—but still, there’s a big “but” here: how much of this translates to other model architectures and ToM tasks beyond false beliefs across a wider range of data? I think its likely other models will use similar mechanisms<sub>[<a href="https://arxiv.org/pdf/2407.10827" title="Tigges" rel="nofollow">25</a>]</sub>, but these are questions not fully answered by this work. 

Further experiments will also explore a wider range of model architectures, the application of crosscoders, transcoders, datasets, and tasks covering more ToM aspects to assess the robustness and generalizability of these findings. For example, it would be interesting to study the difference between predicting why someone will act (ex-ante—predict why a subject will perform an action.) but with more focus on explaining why they did act (ex-post—isolates how the model understands rather than being concerned with its prediction) or develop more empirical studies or experiments that directly test hypotheses about weak grounding or emergent semantics in LLMs, in a broader ToM context.

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

Tigges, *LLM Circuit Analyses Are Consistent Across Training and Scale* Eluether AI Institute, ILLC, University of Amsterdam, Brown University. 2024.[<a href="https://arxiv.org/pdf/2407.10827" title="Tigges" rel="nofollow">16</a>]

Kosinski, *Evaluating Large Language Models in Theory of Mind Tasks.* Stanford University. 2023.[<a href="https://arxiv.org/pdf/2302.02083" title="Kosinski" rel="nofollow">17</a>]

Bricken, *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* Anthropic. 2023.[<a href="https://transformer-circuits.pub/2023/monosemantic-features/index.html" title="Bricken" rel="nofollow">18</a>]

Templeton, *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Anthropic. 2024.[<a href="https://transformer-circuits.pub/2024/scaling-monosemanticity/" title="Templeton" rel="nofollow">19</a>]

Bills, *Language models can explain neurons in language models* OpenAI. 2023.[<a href="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain" title="Bills" rel="nofollow">20</a>]

Li, *The Geometry of Concepts: Sparse Autoencoder Feature Structure* MIT. 2024.[<a href="https://arxiv.org/html/2410.19750v1" title="Li" rel="nofollow">21</a>]

McDougall, *Copy Suppression: Comphrehensively Understanding an Attention Head.* Independent, University of Texas, Google Deepmind. 2023.[<a href="https://arxiv.org/pdf/2310.04625" title="McDougall" rel="nofollow">22</a>]

Jamali, *Unveiling theory of mind in large language models: A parallel to single neurons in the human brain.* Massachusetts General Hospital, Harvard Medical School, MIT. 2023.[<a href="https://arxiv.org/pdf/2309.01660" title="Jamali" rel="nofollow">23</a>]

Bender, *Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data.* University of Washington, Saarland University. 2020.[<a href="https://aclanthology.org/2020.acl-main.463.pdf" title="Bender" rel="nofollow">24</a>]

Tigges, *LLM Circuit Analyses Are Consistent Across Training and Scale.* EleutherAI, ILLC, University of Amsterdam, Brown University. 2024.[<a href="https://arxiv.org/pdf/2407.10827" title="Tigges" rel="nofollow">25</a>]

McDougall, *Indirect Object Identification Exercises and Solutions used in sections 3.2, 3.3, 3.4* Independent. 2024.[<a href="https://colab.research.google.com/drive/1KgrEwvCKdX-8DQ1uSiIuxwIiwzJuQ3Gw?usp=sharing#scrollTo=IzLtmTaNl6mM5" title="McDougall" rel="nofollow">26</a>]

Hardy, *Code for the project can be found here*.[<a href="" title="Hardy" rel="nofollow">27</a>]
