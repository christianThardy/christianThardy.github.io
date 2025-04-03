# The ability to filter information is fundamental to reasoning, but to what degree?

##### Originally posted: 04/02/25

##### tl;dr: 

*Just like your brain filters out irrelevant stuff to focus on, language models use similar "suppression mechanisms" to reason effectively. Analyzing various models (1.3B to 70B parameters), I discovered they all maintain about 50% negative eigenvalues in key attention components, basically meaning half their processing power is dedicated to filtering information rather than amplifying it. This suppression seems crucial for handling complex reasoning tasks like theory of mind (ToM), situational awareness etc., where models need to track multiple perspectives or concepts simultaneously. Different models implement the same mechanism in diverse ways, but the underlying principle appears universal. The ability to selectively suppress information is just as important to reasoning as the ability to highlight it.*

<br>

In cognitive neuroscience, it is widely known that inhibition is a component of the process of selective attention and is manifested in the suppression of goal irrelevant stimuli<sub>[<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC1751480/" title="Dimitrov" rel="nofollow">1</a>]</sub>.

Similar to human inhibition, I think that suppression mechanisms in transformer models constitute a foundational element of reasoning capabilities.

Gradient descent when performed by backpropagation can produce parameter adjustment deltas that are either positive or negative. Positive values or excitatory weights, decrease the weakening of the parameters signal path, which increases the signal strength, and negative values or inhibitory weights increase the weakening of that path, suppressing signal strength through the connection. Thinking about these weights at the attention mechanism, they help determine whether an attention head will become active, not active or suppressed based on the weighted sum of its inputs and threshold depending on the context of the input. So some similarity to the inhibition of a neural pathway is attained through a decrease in the models parameter value. 

In a sense, inhibition in the brain occurs at multiple architectural levels, inside the cell, between cells, and over structures of cells, and the alignment of pulses temporally (in time). Until recently, we can now see this clearly simulated in machine learning constructs, specifically for transformer models, where we see the pathway starting at the loss function, inside the Hessian, between attention mechanisms, and through multi-layer perceptrons.

I recognize I'm anthropromorphizing, and it may not help for general understanding to draw parallels between ReLU activation functions and the functions of synapses with their regional brain chemistry sensitivities and neuronal memory tricks organized by organelles. The comparison is not meant to mislead, but to allow us to appreciate their surface similaries in light of their unique properties. But let's be honest, neuroscience and computational learning models are trapped in an ouroboros of mutual explanation, where each field borrows concepts from the other.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/aae42bfd-b5b8-4abf-bc9c-801e1e16960d" width="400"/>
<br>
<small style="font-size: 12px;">Human brain & shoggoth ouroboros courtesy of GPT-4o.</small>
</p>

<br>

This ouroboros isn't just metaphorical, it reflects genuine functional parallels. For instance, non-linearities exist in both systems, though with crucial differences in implementation. 

While artificial networks use simple activation functions, biological neural circuits exhibit complex non-linearities where potential change isn't just curved, but its function's curvature changes rapidly and is temporally sensitive. This complexity in biological systems hints at why approaches like Hessian curvature analysis might offer valuable insights for deepening our understanding of artificial neural reasoning mechanisms.

Before we continue, I use the term **suppression** across a few related, but distinct computational contexts, which requires clarification. There are four manifestations of suppression I'm interested in:

- Inhibitory weights in neural pathways, kind of an abstract concept in the context I just used it in, borrowed from neuroscience and refers to parameters that dampen signal propagation.
  
- Negative diagonal values in OV circuits, which represent token-specific inhibition, where a head attending to token **X** actively suppresses the probability of that same token appearing in the output.
  
- Negative eigenvalues in OV matrices, which represent directions in the representation space where information is actively flipped or suppressed. While related to negative diagonals, they capture a broader phenomenon where negative diagonals primarily affect self-suppression, while the full spectrum of negative eigenvalues influences how entire subspaces of information are transformed.
  
- Negative eigenvalues in the Hessian, which operate at the meta-level of optimization, representing directions in parameter space where the loss landscape curves downward. These are conceptually distinct from OV eigenvalues but may influence how suppression mechanisms develop during training.

Exluding (1), I think these mechanisms likely interact, but my empirical findings primarily concern (2) and (3), with connections to (4) remaining speculative and requiring further research.

Expanding on my <a href="https://xtian.ai/tom-interpretability#copy-supressions-role" title="ToM in LLMs" rel="nofollow">investigation of suppression</a> by mechanistically studying ToM, analysis of the negative diagonals in OV circuits of key attention heads performing the task across 10 models of parameter counts ranging from 1.3B to 70B, revealed sparse sets of attention heads (~16% on average) that maintain the counterfactual narrative states required to make the correct predictions. Despite architectural differences, all models consistently maintained ~50% negative eigenvalue ratios of the OV matrices (how much suppression is occuring across the observed attention heads), showing a clear progression from concentrated to distributed dimensional processing that scaled with model size, suggesting suppression as a universal computational structure across models. 

Trying to refrain from dubious overgeneralization, its possible the consistent ~50% negative eigenvalue ratio observed in the models studied may manifest differently in larger models. The core principle that likely generalizes is the need for some form of information filtering or suppression mechanism. For instance, recurrent architectures without attention will implement suppression through different mechanisms than the attention-based suppression observed in transformers.

These heads exhibited significantly different suppression patterns across reasoning contexts (93-96%) compared to factual contexts (~65%). Showing a near universal amount of suppression being applied across diverse reasoning modalities ranging from ToM to counterfactual reasoning, goal representation and situational awareness, where each shares key principles:

- Information separation: keeping distinct tracks of information
- Contextual filtering: selective attention to relevant information
- Counterfactual maintenance: suppressing actual world knowledge when needed
- Meta-representation: representing representations themselves
- Temporal sequencing: maintaining ordered steps or states

Looking across various research directions to find anecdotes, I came across state-space models like Mamba<sub>[<a href="https://arxiv.org/pdf/2411.12537" title="Grazzi" rel="nofollow">2</a>]</sub>, and its shown that restricting eigenvalues of state-transition matrices to `[0, 1]` limits their ability to solve even simple tasks like parity, and how extending the eigenvalue range to `[-1, 1]` dramatically improves the expressivity of these models on state-tracking tasks.

We can see further connections between model expressivity and optimization across the loss landscape, and we can reasonably guess this may generalize to other architectures if certain conditions hold.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/da2a160b-ac6f-43c5-8d7e-59ec9a616d57" width="550"/>
<br>
<small style="font-size: 12px;">Figure 2d. "The subspace defined by the negative eigenvalues is less stable than the one defined by the positive ones"<sub>[<a href="https://arxiv.org/pdf/1902.02366" title="Alain" rel="nofollow">3</a>]</sub>.</small>
</p>

<br>

Comparing my results to the analysis of Alain et al., they demonstrate that the top eigenspace (positive curvature) of the Hessian remains relatively stable during training, while the bottom eigenspace (negative curvature) is far less stable, which mirrors my finding that different models implement consistent suppression mechanisms but with varying implementations.

This could roughly explain why I observed diverse suppression strategies across models (Gemma 2-2B's persistent high suppression vs. the Llama family's calibrated moderate suppression vs. Pythia 1.4B's phase-shifted suppression). The varying implementations of suppression could be a direct result of the instability of negative eigenspaces during training, which is maybe why architectures develop different suppression mechanisms that operate the same on similar tasks. 

It's plausible that the instability contributes to *discovered* behaviors, so the network discovers viable suppression strategies through stochastic gradient interactions or phase changes during training rather than converging to a single canonical implementation.

In my analysis, the Llama family of models 3B, 8B and 70B, showed consistency in their "suppression signatures" despite model size, suggesting that architectural inductive biases create basins of attraction for specific suppression strategies despite the instability of negative eigenspaces. I'm pretty bullish on training dynamics determining which specific strategies emerge within architectural constraints based on the training data and the interaction between architecture and training creating characteristic suppression signatures.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/69865b11-614d-40bc-b971-017166333166" width="850"/>
<br>
<small style="font-size: 12px;">Figure 5. "We also do not have α*_i ≈ 1/|λ_i| but rather the optimal stepsize seems to be decorrelated from the eigenvalue"</small>
</p>

<br>

They also find that while the optimal step size in directions of positive curvature follows the expected α* = 1/λ relationship, this completely breaks down for directions of negative curvature, which *double dubiously* parallels my find that suppression mechanisms don't operate as simple inverses of amplification mechanisms. The decorrelation between eigenvalue magnitude and optimal behavior in those directions might suggest that suppression is fundamentally different from amplification.

The complexity of negative eigenspaces might explain the sophisticated, multi-level suppression flow I identified in the <a href="https://xtian.ai/tom-interpretability#tom-circuit" title="ToM in LLMs" rel="nofollow">ToM circuit</a>. This finding basically tells us that what's happening is non-linear, which could explain why suppression mechanisms might need to be implemented through sophisticated, coordinated patterns across multiple components rather than simple magnitude-based relationships, as simple linear suppression would be insufficient for tracking complex, nested belief states, counterfactual states or multiple concepts at once or sometimes separate or parallel representations, which enables the selective, context-dependent information filtering required for reasoning.

This would also explain my observation in the ToM analysis that *the model doesn't "represent John's beliefs" so much as it systematically prevents certain information pathways from overwriting others*; suppression creates functional separation through complex, distributed mechanisms rather than explicit representational schemes.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/71cb09ac-2898-4a29-ab7d-b8587f8840b1" width="850"/>
<br>
<small style="font-size: 12px;">Figure 6. "Shows that the most improvement is obtained when optimizing in the directions of negative curvature"</small>
</p>

<br>

The paper demonstrates that directions of negative curvature often contain significant potential for loss improvement. Seems pretty obvious, but supports my hypothesis that suppression mechanisms play a crucial role in enabling reasoning capabilities. In optimization terms, moving along directions of negative curvature can produce larger decreases in the loss function during training than moving along directions of positive curvature because positive curvature directions saturate. So a simple explanation would be that models actively develop suppression mechanisms because they significantly reduce loss in reasoning tasks that require maintaining multiple representational states like ToM or counterfactual reasoning. This is why selective attention is so crucial.

My next steps are to establish causality beyond correlation, so I'll need comphrehensive interventional studies. After that I can see interesting aignment relevant investigations coming from this work in the form of behavioral detection, mechanism steering, novel loss functions or advanced low-rank updates to mechanisms.

But an important limitation of my analysis is the gap between Hessian negative eigenvalues, which operate at the level of the entire loss landscape, and negative diagonal eigenvalues of the OV circuit, which function at the level of attention mechanisms. While both involve negative eigenvalues associated with suppression, they operate at different levels of abstraction. The Hessian eigenvalues influence how parameters update during training, potentially shaping how OV circuits develop their suppression characteristics. While this relationship is plausible, it remains largely theoretical to me, but I can't wait to test it. 

Another area of research that I'm super excited about through the lens of interpretability and alignment is <a href="https://docs.google.com/document/d/1DwO_Bpoqfe3F0LhNDMtNUqidlS3K_M78sWqbD7J1up0/edit?pli=1&tab=t.0" title="google doc" rel="nofollow">singular-learning theory</a>, which could shed light on *what* suppression strategies emerge and explain *why* they emerge and predict *which* strategies will develop under various conditions, which would transform my descriptive analysis into a predictive theory of suppression mechanism development.

<br>

### References:

Dimitrov, *Inhibitory attentional control in patients with frontal lobe damage.* Brain Cogn. 2003.[<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC1751480/" title="Dimitrov" rel="nofollow">1</a>]

Grazzi, *Unlocking State-Tracking In Linear Rnns Through Negative Eigenvalues.* CSML, Istituto Italiano di Tecnologia, University of Freiburg, ELLIS Institute Tubingen, AI Centre, University College London. 2025.[<a href="https://arxiv.org/pdf/2411.12537" title="Grazzi" rel="nofollow">2</a>]

Alain, *Negative eigenvalues of the Hessian in deep neural networks* Mila, University of Montreal, Google Brain. 2019.[<a href="https://arxiv.org/pdf/1902.02366" title="Mahowald" rel="nofollow">3</a>]
