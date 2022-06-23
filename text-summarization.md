# text summarization on complex data using transformers

Research and development have very broad implications. After all, we're trying to push the SOTA in our field by looking for hard problems to solve that can make a huge impact. Solving them allows us to redefine the way we look at our work, and shape the future of our organization. 

About 2 to 3 years ago extractive summary algorithms, graph based reduction summarization and importance ranking based on engineered features were pretty popular for 
document summarization. But what do you use when you have a lot of nuanced conditions? What do you do when you want to distill a complex piece of text down to its most basic idea, summarize the most interesting parts where specific entities are present and you want control over the context of the output?

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/175053632-8534d9fe-b5b6-4737-a627-350d57254fb3.PNG" width="455px"></b><br>
</p>

As humans, we can profit from the experience of someone older or someone who has more experience in what it is we're trying to acheive. Analogous to this, we can think 
of transfer learning as reusing a model developed for a task as the starting point for a model on a second task. The architecture I fine-tuned and used for our model is called a transformer, specifically the T5 (text-to-text transfer transformer), which takes text as input and generates some target text. 

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/175078481-54b16b89-f9c4-4008-8b5d-d55fc2be0132.gif" width="455px"></b><br>
</p>

Yesteryear, sequencing tasks in natural language processing were dominated by autoencoders(seq2seq), recurrent neural networks and convolutional neural networks. 
You could build an attention mechanism into these architectures, but so far attention is the main component needed to solve a lot of long range dependency 
problems in NLP. 

While rnns tend to forget the words they learn over time, and cnns suffer from the need of a ridiculous amount of layers without the promise of convergence, the 
transformers stacked self-attention layers allow them to see different positions of words to compute a representation of sets of words. This allows them to encode
more <a href="https://user-images.githubusercontent.com/29679899/104795121-fc456e00-5779-11eb-8126-2bcd5cec0152.png" title="Yoshua Bengio's thoughts on the subject" rel="nofollow">compositional</a> information than any model before them. This is huge, just ask the engineer who said <a href="https://www.giantfreakinrobot.com/tech/artificial-intelligence-hires-lawyer.html" title="Can't tell if this is cap or not" rel="nofollow">Google's new question answering system is sentient</a>. As one of the main learning components for this system, transformers could very well be at the forefront of what it means to create general intelligence. <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html" title="This is definitely not cap" rel="nofollow">NLP has really taken off this year 😬</a>.

<br/>

## the data

You have simple text and then you have hard text. With simple text, you can use simple parsing methods to solve easy to intermediate problems. With hard text you
can try simple and intermediate methods, but there are no guarantees. We can think of hard text as something it would take a subject matter expert to understand. For
example, the abstract of a paper:

*Systems and processes for rule-based natural language processing are provided. In accordance with one example, a method includes, at an electronic device with one or more processors, receiving a natural-language input; determining, based on the natural-language input, an input expression pattern; determining whether the input expression pattern matches a respective expression pattern of each of a plurality of intent definitions; and in accordance with a determination that the input expression pattern matches an expression pattern of an intent definition of the plurality of intent definitions: selecting an intent definition of the plurality of intent definitions having an expression pattern matching the input expression pattern; performing a task associated with the selected intent definition; and outputting an output indicating whether the task was performed.*

Documents like this can be exhaustively long, and sometimes words can have multiple meanings which translates to amgiuous and less common parts-of-speech. 

For instance *“said”* most commonly functions as an adjective, and *“claim”* typically functions as a noun when occurring in text like this, but these words more typically appear as verbs in news articles, the web, and other sources of text which off the shelf models are usually trained on. This misidentification misidentifies other words compositionally, which substantially damages the ability to construct consistent compositionality and can lead to incorrect predictions, which is why text like this can be hard to work with. 

To summarize the above paragraph, you could use something off of the shelf so to speak, but then you can expect off the shelf results...

Google's Pegasus model trained on the big patent dataset would summarize the paragraph as: 

#### *"Rule-based natural language processing is provided."*

<br/>

Vanilla extractive summarizer: 

#### *"Systems and processes for rule-based natural language processing are provided. In accordance with one example, a method includes, at an electronic device with one or more processors, receiving a natural-language input; determining whether the input expression pattern matches a respective expression pattern of each of a plurality of intent definitions; performing a task associated with the selected intent definition; and outputting an output indicating whether the task was performed."*

<br/>

Vanilla abstractive summarizer: 

#### *"Computer scientists at the Massachusetts Institute of Technology (MIT) have developed a method for processing natural language."*

The output from the 1st and 3rd models meet our complex to simple heuristic, a few entities of interest are present to capture a bit of context, but neither is very interesting, the 2nd model repeats everything verbatim, while the 3rd model is hallucinating facts that do not exist.

<br/>

# to be continued...

<br/>
<br/>

### references:

Roberts, Raffel, *Google AI Blog: Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer.* Google Research. 2020.[<a href="https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html" title="Transfer Learning" rel="nofollow">1</a>]
