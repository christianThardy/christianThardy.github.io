# two realities

Paradoxically, mathematical reality lies outside of us. Our function is to discover and observe it.

I will use the noun intuition to denote the ability to understand an idea without the need for conscious reasoning.

The adjective real in the same sense that you can think about the material world with respect to day and night, or a <a href="https://www.space.com/43062-super-blood-moon-2019-last-until-2021.html" title="space.com" rel="nofollow">super blood wolf moon</a>. It is outside, independent of us.

I will define the noun reality with two separate connotations, physical reality and artificial reality. 

By using the adjective physical I mean the reality that you and I inhabit. The same reality that physicists try to describe using mathematics. 

Then we have the adjective artificial which will be referred to in its ordinary sense, *"Made or produced by human beings rather than occurring naturally, especially as a copy of something natural"*.

The connotations between the physical and artificial states ring true in the world of machine learning optimization, in the sense that the physical and artificial realities coexist. 

The role of visualization in data science can be a very broad topic[<a href="https://www.semanticscholar.org/paper/Seeing-is-believing%3A-The-importance-of-in-machine-Vellido-Mart%C3%ADn-Guerrero/e692853acaab3221fe92ccb9f6e06f651ce67064?p2df" title="Visualizations & machine learning research paper" rel="nofollow">1</a>], usually lending itself to a graph that explains a descriptive business metric...

<br/>

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/101286427-7a484980-37b8-11eb-9450-98fba3bb3833.jpg" width="445px"></b><br>
</p>

<br/>

...or some goal centered around a base of statistics relevant to a business problem that will influence a decision making process.

<br/>

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/101286464-aa8fe800-37b8-11eb-945a-9b417f28d3c1.jpg" width="445px"></b><br>
</p>

<br/>

But I find that in a machine learning research environment, plotting something like the convexity or non-convexity of a loss function...

<br/>

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/101286407-597ff400-37b8-11eb-88f3-3606642eddf3.jpg" width="445px"></b><br>
</p>

<br/>

...or the hidden representation of a neural network trying to learn a predictive classification task...

<br/>

<p align="center">
  <img src="https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/topology_2D-2D_train.gif" width="345px">
</p>

<br/>

...serves a different purpose. It is not a true representation of anything in the physical world. It is a virtual, or simulated reality used to understand the abstract nature of a concept, theorem, data, or even a reinforcement learning agent. 

Metaphorically speaking, the function of a visualization exists to stimulate the imagination. They literally aid in the ability to understand concepts both practical and abstract, but the truth of a theorem is not affected by the quality of the visualization, as it is a tool to make the meaning of the function easier to understand. This is to say in some sense, that what you see is not real.

For example, let's look to the major branch of mathematics known as topology. It has many practical applications that help us solve real non-trivial problems, such as the mobius strip[<a href="https://mathworld.wolfram.com/MoebiusStrip.html" title="aylien.com" rel="nofollow">2</a>]...

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/101286901-36a30f00-37bb-11eb-92af-825516fae8c6.jpg" width="445px">
</p>

<br/>

...being used as a belt for a conveyor in the 1950s.

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/101287051-fc863d00-37bb-11eb-95e9-5ac908399094.gif" width="445px">
</p>

<br/>

The belt being used for the conveyer is a part of the real world. Suppose the conveyor is being used in some sort of mining task where the typical conveyer belt lifespan is dependent on top cover wear and cut damage. The mobius strip belt handles this elegantly, as the middle half twist in the belt allows the belts surface area to wear equally.

Let's say the belt for whatever reason breaks. The theorem defining the existence of the mobius strip, does it change because its physical representation is destroyed? The mobius strip exists independently of the belt whose design is based on the theory and is further independent of any other detail of the real world. The inverse is also true.

As illustrated in the gif above, topology applied to data and neural networks is a very abstract model as opposed to its representation of physical reality. For example, the concrete application of homeomorphisms as applied to problems concerning large data sets, tries to describe global features of a space dependent on the data contained within the space locally[<a href="http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/" title="Chris Olah's blog" rel="nofollow">3</a>].

Given the connection between data, neural networks and topology, visualizations like the gif above open a door in helping us to understand what a neural network is really doing when its trying to approximate a representation. This helps to deepen our intuition regarding feature spaces, high/low dimensional qualities of data and latent structures in large datasets[<a href="https://www2.math.upenn.edu/~ghrist/preprints/nieuwarchief.pdf" title="Homology research paper" rel="nofollow">4</a>]. 

In the absence of a visualization or intuitive illustration, describing something like gradient descent would be tricky. As its visual, physical representation are just real numbers starting with some very high value and descending through time until it reaches some optimal value...

<p align="center">
  <b><img src = "https://user-images.githubusercontent.com/29679899/101287170-9ea62500-37bc-11eb-89dc-9a7aa19de85c.gif" width="480"></b><br>
</p>

A number of popular illustrations regarding this process denotes a mountain (parameter space) by which someone (a gradient) must descend (find the location in the parameter space where the badness of fit is minimized and the goodness of fit is maximized) and the time it takes her to get to the bottom (where the function fits the data best). 

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/101287186-b7163f80-37bc-11eb-843e-7c7f8b140e2a.PNG" width="350px">
</p>

<br/>

This illustration of gradient descent is important to us because of the parallel drawn from its physical representation and it allows us to gain insight into the idea without the frame of conjecture. This simple pose allows the intuition for gradient descent to take <a href="http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html" title="Alec Radford's blog" rel="nofollow">many</a> forms:

<br/>

<p align="center">
  <img src = "https://user-images.githubusercontent.com/29679899/101287297-6e12bb00-37bd-11eb-88c9-da076b596e4c.gif" width="500px">
</p>

<br/>

These philosophical implications even ring true for the hairy ball theorem[<a href="https://youtu.be/B4UGZEjG02s" title="The hairy ball theorem" rel="nofollow">'</a>].

As I try to organize some of my high level thoughts on the role that  visualization plays in understanding how computers learn, one thing is certain. The rich connection between distilling data into something useful and trying to understand the abstractions of learning machines is something that will continually spark my enthusiasm for artificial intelligence. 

Things like neural ordinary differential equations, optimization algorithm development, statistical learning methods, machine learning interpretability, and safety are some of the topics at the forefront of researchers minds today, and tied to the mathematics and theory of every paper will be a <a href="https://imgur.com/2pyn9sU" title="ODE Solvers as machine learning optimizers" rel="nofollow">map</a> leading us from the abstract to something that we can understand.

<br/>

## References:

Vellido, Martin. Rossi, Lisboa, Seeing is Believing: The Importance of Visualization in real-world machine learning applications. 2011.[1]

Olah, Neural Networks, Manifolds and Topology. 2014.[2]

Ghrist, Three Examples of Applied and Computational Homology. 2008.[3]

Mahadevan, Imagination Machines: New Horizons in Artificial Intelligence. 2018.[4]
