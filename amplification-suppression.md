# The ability to filter information is fundamental to reasoning

<br> 

**Suppression and inhibition are key to emergent behavior, but to what degree**

Analogous to human inhibition, mechanisms that replicate this function in transformer models could loosely described as the bedrock for their reasoning. Spiking neural networks have always escaped my understanding as the type of biological mechanisms they try to mimick are already normally designed in and are implemented by neural networks already.

Gradient descent when performed by back-propagation can produce parameter adjustment deltas that are either positive or negative. Positive values or excitatory weights, decrease the weakening of the parameters signal path, which increases the signal strength, and negative values or inhibitory weights increase the weakening of that path, suppressing signal strength through the connection. Thinking about these weights at the attention mechanism, they help determine whether an attention head will become active, not active or suppressed based on the weighted sum of its inputs and threshold.

Some similarity to the inhibition of a neural pathway is attained through a decrease in the parameters value. In a sense, inhibition in the brain occurs at multiple architectural levels, inside the cell, between cells, and over structures of cells, and the alignment of pulses temporally (in the time domain) and until recently, was not simulated at all in multiple machine learning architecture constructs, until the transformer model, where we see the pathway starting at the loss function, inside the Hessian, between attention mechanisms, and through the multi-layer perceptrons.

I'm engaging in a lot of anthropromorphization, yes, and it may not be an aid to general understanding to draw parallels between ReLU activation functions and the functions of synapses with their sensitivity to regional brain chemistry and with cell-level retention functions orchestrated by organelles. It's not my fault neuroscience and any computational model of how machines learn are ouroboros-like in their nature, where methods from one can be borrowed to explain the other. 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/aae42bfd-b5b8-4abf-bc9c-801e1e16960d" width="400"/>
<br>
<small style="font-size: 12px;">Human brain & shoggoth ouroboros courtesy of ChatGPT-4o.</a></small>
</p>

<br>

Nonetheless, there are non-linearities of different types in biological neural circuits. Potential change is not only curved, but its function's curvature changes quickly. It is temporally sensitive. Hessian curvature analysis offers several insights that align with and could deepen interpretability insights in reasoning.

Expanding the theory of mind (ToM) research, analysis of the negative diagonals of OV matrices of key attention heads, across 10 models of parameter counts ranging from 1.3B to 70B revealed sparse sets of attention heads (~16% on average) that maintain counterfactual narrative states. Despite architectural differences, all models consistently maintained ~50% negative eigenvalue ratios (how much suppression is occuring) in observed attention heads, showing a clear progression from concentrated to distributed dimensional processing that scaled with model size, suggesting suppression as a universal computational structure across models. 

These heads exhibited significantly different suppression patterns across reasoning contexts (93-96%) compared to factual contexts (~65%). Showing a near universal amount of suppression being applied across diverse reasoning modalities ranging from ToM to counterfactual reasoning, goal representation and situational awareness, where each shares key principles:

- Information separation (keeping distinct tracks of information)
- Contextual filtering (selective attention to relevant information)
- Counterfactual maintenance (suppressing actual world knowledge when needed)
- Meta-representation (representing representations themselves)
- Temporal sequencing (maintaining ordered steps or states)

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/da2a160b-ac6f-43c5-8d7e-59ec9a616d57" width="550"/>
<br>
<small style="font-size: 12px;">Figure 2d. The subspace defined by the negative eigenvalues is less stable than the one defined by the positive ones<sub>[<a href="https://arxiv.org/pdf/1902.02366" title="Mahowald" rel="nofollow">1</a>]</sub>.</a></small>
</p>

<br>

Comparing my results to the analysis of Alain et al., they demonstrate that the top eigenspace (positive curvature) of the Hessian remains relatively stable during training, while the bottom eigenspace (negative curvature) is far less stable, which mirrors my finding that different models implement consistent suppression mechanisms but with varying implementations.

This could roughly explain why I observed diverse suppression strategies across models (Gemma's persistent high suppression vs. Llama's calibrated moderate suppression vs. Pythia's phase-shifted suppression). The varying implementations of suppression could be a direct result of the instability of negative eigenspaces during training, which is maybe why architectures develop different suppression mechanisms despite solving the same tasks. 

It could be plausible that the instability contributes to "discovered" rather than "designed" behaviors, so the network discovers viable suppression strategies through stochastic gradient interactions or phase changes during training rather than converging to a single canonical implementation.

But three specific models in my analysis, Llama 3B, 8B and 70B, showed consistency in their "suppression signatures" despite model size, suggesting that architectural inductive biases create basins of attraction for specific suppression strategies despite the instability of negative eigenspaces. So its possible that training dynamics determine which specific strategy emerges within architectural constraints and the interaction between architecture and training creates characteristic suppression signatures.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/69865b11-614d-40bc-b971-017166333166" width="850"/>
<br>
<small style="font-size: 12px;">Figure 5. "We also do not have α*_i ≈ 1/|λ_i| but rather the optimal stepsize seems to be decorrelated from the eigenvalue."</a></small>
</p>

<br>

They also find that while the optimal step size in directions of positive curvature follows the expected α* = 1/λ relationship, this completely breaks down for directions of negative curvature, which *dubiously* parallels my find that suppression mechanisms don't operate as simple inverses of amplification mechanisms. The decorrelation between eigenvalue magnitude and optimal behavior in those directions might suggest that suppression is fundamentally different from amplification.

The complexity of negative eigenspaces might explain the sophisticated, multi-level suppression flow I identified in the ToM circuit. This finding basically tells us that what's happening is non-linear, which could explain why suppression mechanisms might need to be implemented through sophisticated, coordinated patterns across multiple components rather than simple magnitude-based relationships, as simple linear suppression would be insufficient for tracking complex, nested belief states, counterfactual states or multiple concepts at once or sometimes separate or parallel representations, which enables the selective, context-dependent information filtering required for reasoning.

Which explains my observation in the ToM analysis that "the model doesn't 'represent John's beliefs' so much as it systematically prevents certain information pathways from overwriting others"; suppression creates functional separation through complex, distributed mechanisms rather than explicit representational schemes.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/71cb09ac-2898-4a29-ab7d-b8587f8840b1" width="850"/>
<br>
<small style="font-size: 12px;">Figure 6. "Shows that the most improvement is obtained when optimizing in the directions of negative curvature."</a></small>
</p>

<br>

The paper demonstrates that directions of negative curvature often contain significant potential for loss improvement, supporting my hypothesis that suppression mechanisms play a crucial role in enabling reasoning capabilities. We can also see that directions of negative curvature often contain significant potential for loss improvement. In optimization terms, moving along directions of negative curvature can produce larger decreases in the loss function than moving along directions of positive curvature because positive curvature directions saturate, and models actively develop suppression mechanisms because they significantly reduce loss. 

So if negative eigenvalues (suppression) contain significant potential for performance improvement, it explains why models develop sophisticated suppression mechanisms for complex reasoning tasks. For reasoning tasks like ToM or counterfactual reasoning, the ability to selectively suppress certain information pathways might be precisely what enable models to maintain separate states for different entities/concepts. The fact that directions of negative curvature retain improvement potential throughout training (unlike positive curvature directions) might explain why suppression mechanisms remain active and important even in fully trained models, showing that suppression is not just a "side effect" of training but a fundamental computational mechanism that actively contributes to reasoning capabilities.

An area of research that I'm super excited about through the lens of interpretability and alignment, singular-learning theory, could shed light on *what* suppression strategies emerge and explain *why* they emerge and predicting *which* strategies will develop under various conditions, which will transform my descriptive analysis into a predictive theory of suppression mechanism development.

<br>

### References:

