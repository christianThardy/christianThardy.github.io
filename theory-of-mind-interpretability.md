# Theory of mind and GPT models

Mechanistic interpretability allows us to reverse engineer the inner workings and representations learned by neural networks into understandable algorithms and concepts that provide a granular, causal understanding of neural networks.

Given my current focus on LLMs and my interest in psychology, I've been asking myself how do decoder-only language models perform theory of mind tasks. I have a theory that some simplification of abstract reasoning tasks like the theory of mind (ToM) task can be interpreted from the inner mechanisms of a GPT model to understand its internal representations of ToM tasks. If the circuit (algorithm) that completes this task can be reverse engineered, what makes that possible in a GPT-2 model?

Humans are capable of making inferences about the mental state of characters in a ToM sentence. These inferences require syntactic or prepositional logic, but what else? Let's first explore the linguistic phenomena of **First-Order Logic** (FOL), **Semantics** and **Pragmatics**.

<br>

# First-Order Logic

Sentences where you can make inferences require FOL, semantics and pragmatics. It provides a framework for representing and manipulating the meaning of sentences in a structured and formal way, also helps in mapping syntactic structures of natural language sentences to their corresponding semantic representations.

Let's take an example sentence: *Today is Jane's birthday. She looked at the empty plate and sighed. She thought, 'If only John had remembered to bring the...'*

In the context of ToM, to make the correct prediction *cake*, the model needs to understand:

  - **Entities:** Jane, John, the empty plate.

  - **Properties and Relations:** It is Jane's birthday, Jane's expectation about John bringing something, the state of the plate being empty, Jane's sigh indicating disappointment.

  - **Mental States:** Jane's belief and expectation that John would bring the cake to her birthday.

<br>
  
FOL helps in maintaining the context and managing the state of a conversation by representing a dialogue state in logical terms. For example:

  - Sigh(Jane)\
    Empty(Plate)\
    Birthday(cake)\
    Bring(John,Cake)\
    Expect(Jane,John,Bring(John,Cake))

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/fa03f9ec-4f91-4c8a-82f4-d6a30cdea719">
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
      - Jane's actions and mental state: Sigh(Jane), Think(Jane,ifonlyJohnhadrememberedtobringthecake)

<br>

In the context of semantics, ToM prediction requires extracting the meaning of a sentence, including understanding entities, their properties, and relationships, which is the core goal of semantic parsing. Semantic parsing can help with understanding context and inferring implied meanings, which is essential for accurate ToM predictions. ToM prediction also involves understanding and representing complex mental states and expectations, which require a structured form that semantic parsing provides. LLMs can understand the underlying meaning and context, allowing them to predict that the missing word is *cake*. This involves both understanding the literal content and inferring the mental states and expectations of the characters.

<br>

# Pragmatics

Pragmatics, usually a key concept in semantics, is focused on how context influences the interpretation of meaning in language. This includes factors like speaker intent, conversational implicature, and situational context. To predict the final word in the example sentence sequence, a model must understand not just the literal meaning of the words but also Jane's mental state, her expectations, and the context in which she is making the statement.

To obtain contextual understanding we need to know situational context, so an empty plate, a sigh, it being Jane's birthday helps infer that something was expected on the plate on this day. A speakers intention and beliefs, so understanding Jane’s disappointment and what she believes John was supposed to bring, and we need the ability to infer the most likely item that fits Jane’s expectation and the context (e.g., a specific food item like "cake").

ToM prediction heavily relies on the context to make sense of the mental states and intentions behind the words, and the final word prediction is based on implied meanings and inferred intentions, which are central to pragmatics. Pragmatics encompasses understanding social interactions, cognitive states, understanding that others have mental states, beliefs, desires, intentions, and perspectives—that are different from one's own, which are key to ToM.

The remainder of this work will specifically focus on how GPT models will implement this task and in the end understand in a tractable way, the mechanisms responsible for completing the task across many different heuristics and metrics.

<br>

# 
