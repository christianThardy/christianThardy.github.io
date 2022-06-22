I love contributing to research and development, because the work has very broad implications. After all, we're trying to push the SOTA in our field by looking for hard 
problems to solve that can make the biggest impact possible to redefine the way we look at our work, and to shape optimizations to come. 

About 2 to 3 years ago extractive summary algorithms, graph based reduction summarization and importance ranking based on engineered features were pretty popular for 
document summarization. But what do you use when you have a lot of nuanced conditions? You want to distill complex data down to its most basic idea, summarize key 
information in a document that's interesting for a specific purpose, and you want control over the context of the output?

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/175053632-8534d9fe-b5b6-4737-a627-350d57254fb3.PNG" width="455px"></b><br>
</p>

As humans, we can profit from the experience of someone older or someone who has more experience in what it is we're trying to acheive. Analogous to this, we can think 
of transfer learning as reusing a model developed for a task as the starting point for a model on a second task. The architecture I fine-tuned and used for our model is 
called a transformer, specifically the T5 (text-to-text transfer transformer), which takes text as input and generates some target text. 

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/175078481-54b16b89-f9c4-4008-8b5d-d55fc2be0132.gif" width="455px"></b><br>
</p>

Yesteryear, sequencing tasks in natural language processing were dominated by autoencoders(seq2seq), recurrent neural networks and convolutional neural networks. 
You could build an attention mechanism into these architectures, but so far attention is the main component necessary to solve a lot of long range dependency 
problems in NLP. 

While rnns tend to forget the words they learn over time, and cnns suffer from the need of a ridiculous amount of layers without the promise of convergence, the 
transformers stacked self-attention layers allow them to see different positions of words to compute a representation of sets of words. This is huge,
just ask the engineer who said <a href="https://www.giantfreakinrobot.com/tech/artificial-intelligence-hires-lawyer.html" title="Can't tell if this is cap or not" rel="nofollow">Google's new question answering framework is sentient</a>. 
As one of the main learning components for this system, transformers could very well be at the forefront of what it means to create general intelligence. <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html" title="This is definitely not cap" rel="nofollow">NLP has really taken off this year 😀</a>.

<br/>

# to be continued...

<br/>
<br/>

### references:

https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
