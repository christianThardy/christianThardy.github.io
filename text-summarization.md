# text summarization on complex data using transformers

Summaries help us extract and compress information quickly. If we're summarizing something we read in the morning paper to a friend, or telling them about an experience we had on vacation, pulling out the main points of an event and describing it in our own words is something we do everyday.

The machine learning task of summarization is to convert long text into short text. To do this we need to compute the probability of the text itself to get a probability of word sequences. Good language models allow us to generate summaries that are fluent, and to do this we must also compute the probability of the output text.

About 2 to 3 years ago extractive summary algorithms, graph based reduction summarization and importance ranking based on engineered features were pretty popular for document summarization, but did little to give us control over the context of the predicted result, let alone handle very nuanced conditions; they could not distill complex text down to their most basic ideas and summarize the most interesting parts where specific entities are present in a way that makes it easy for someone reading the summary to know what the document is about. 

These older methods are basically compressors, all they can essentially do are delete tokens from sentences. This presents a problem of alignment, where labeled examples provide the data and the text, but they do not specify which parts of the text correspond to which parts of the data that are interesting for our preferences. 

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/179746579-26034750-dfa6-47f8-ba12-00fa16bffac1.jpg" width="455px"></b><br>
</p>

For my problem, I needed to make a few tweaks architecturally, but the only feature I needed to engineer to shorten a document while preserving its meaning was a small amount of input text and handwritten summaries to generate predicted target text thanks to the T5 (text-to-text transfer transformer).

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/175078481-54b16b89-f9c4-4008-8b5d-d55fc2be0132.gif" width="455px"></b><br>
</p>

As humans, we can profit from the experience of someone older or someone who has more experience in what it is we're trying to acheive. Analogous to this, we can think of transfer learning as reusing a model developed for a task as the starting point for a model on a second task. We assume that a lot of the factors that explain the variations in model A are relevant to the variations that need to be captured in model B, which in most cases will eliminate the need for exhaustive sets of rules and ranking algorithms.

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/175053632-8534d9fe-b5b6-4737-a627-350d57254fb3.PNG" width="455px"></b><br>
</p>

More recently, sequencing tasks in natural language processing were dominated by vanilla autoencoders(seq2seq), recurrent neural networks and convolutional neural networks. 
These methods certainly worked, but for one reason or another they all fell short.

While seq2seq models perform better on shorter text, rnns tend to forget the words they learn over time, and in terms of text summarization cnns suffer from the need of a ridiculous amount of layers without the promise of convergence, the transformers stacked self-attention layers allow them to see different positions of words to compute a representation of sets of words, which allows them to solve a lot of long range dependency problems. In other words, attention can link each part of the generated text back to a record in the data. Not to mention transformers can scale dramatically compared to the other three. You can build an attention mechanism into seq2seq, rnn and cnn architectures, but *self attention* is the main component needed to avoid these pitfalls. 

The way transformers compute over sets of words allows them to encode more <a href="https://user-images.githubusercontent.com/29679899/104795121-fc456e00-5779-11eb-8126-2bcd5cec0152.png" title="Yoshua Bengio's thoughts on the subject" rel="nofollow">compositional</a> information than any model before them. This is huge, just ask the engineer who said <a href="https://www.giantfreakinrobot.com/tech/artificial-intelligence-hires-lawyer.html" title="Can't tell if this is cap or not" rel="nofollow">Google's new question answering system is sentient</a>. As one of the main learning components for this system, transformers could very well be at the forefront of what it means to create general intelligence. <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html" title="This is definitely not cap" rel="nofollow">NLP has really taken off this year 😬</a>.

Recently I fine-tuned the T5 for a summarization task, but before we dive deeper into transformers and this architecture specifically, let's understand the type of model I wanted to create. 

<br/>

## the data

You have simple text and then you have hard text. With simple text, you can use simple parsing methods to solve easy to intermediate problems. With hard text you
can try simple and intermediate methods, but there are no guarantees. 

Let's define *hard text* as documents that can easily lead to information overload because they explain concepts using broad descriptions, and use technical and specific terms that require specialized knowledge to understand the document. For example, this section of a paper:

*The object is achieved by a magnet core with high control and linear BH loops at alternating current and direct current, said magnet core having a relative permeability (μ) above 500 and an amount of magnetostrictive saturation (λ s ) of less than 15 ppm. The method of claim 16, Wherein said heat treatment occurs in the transverse field after heat treatment in the longitudinal field. As a current transformer for alternating current having a magnet core according to claim 1 or 2, In addition to the magnetic core, the current transformer has a primary winding and at least one secondary winding, wherein the secondary winding is terminated to low-resistance by the load resistance and / or the measurement electronics.*

Documents like this can be exhaustively long, and sometimes words can have multiple meanings which translates to amgiuous and less common parts-of-speech. 

For instance *“said”* most commonly functions as an adjective, and *“claim”* typically functions as a noun when occurring in text like this, but these words more typically appear as verbs in news articles, the web, and other sources of text which off the shelf models are usually trained on. This difference in the use of language can potentially misidentify surrounding words compositionally, which substantially damages the ability to construct consistent compositionality and can lead to incorrect predictions, which is why text like this can be hard to work with. 

The goal is to save users a ton of reading time so they can allocate their time more effectively. Ideally for my use-case I would want a summary that tells me in some approachable way that ```transformers``` have issues with ```load resistance``` because of ```secondary magnetic core windings```.  Lets try summarizing the paragraph using existing algorithms from packages trained on out of domain data...

<br/>

Google's Pegasus model trained on the big patent dataset would summarize the paragraph as: 

#### *"A magnet core with high control and linear BH loops at alternating current and direct current, said magnet core having a relative permeability above 500 and an amount of magnetostrictive saturation of less than 15 ppm."*

<br/>

Vanilla extractive summarizer: 

#### *"The object is achieved by a magnet core with high control and linear BH loops at alternating current and direct current. In addition to the magnetic core, the current transformer has a primary winding and at least one secondary winding. The secondary winding is terminated to low-resistance by the load resistance and / or the measurement electronics."*

<br/>

Vanilla abstractive summarizer: 

#### *"The current transformer is one of the world's most powerful current transformers, and has been described as "the most powerful transformer of its kind in the world"*

<br/>

None of them are close to our expected output. The 1st and 3rd models sort of meet our complex to simple heuristic, a few entities of interest are present to capture a bit of context, but neither are very interesting, and the 2nd model repeats everything verbatim. Abstractive summarization should introduce new words or phrases, as I'm after a model that removes technical language and replaces it with more novel, approachable language, but the 3rd model just hallucinates facts that do not exist.

When thinking about this we first need to understand our source documents, compare them to T5's training sources, and handwrite short, coherent, abbreviated versions of our source documents so that they contain the most relevant information we would like to capture. 

<br/>

# to be continued...

<br/>
<br/>

### references:

Roberts, Raffel, *Google AI Blog: Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer.* Google Research. 2020.[<a href="https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html" title="Transfer Learning" rel="nofollow">1</a>]
