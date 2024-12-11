# ToM project notes

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/a1f01bda-0b0b-4ce8-b6ec-51fe0d3ad4b2" width="1000">
</p>

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/68336ba9-9b1f-411d-8963-0daff98a26d4" width="1000">
</p>

<br/>

<p align="center">
  <img src = "https://github.com/user-attachments/assets/33f89aa5-b44d-43f9-82e3-a1aa99819492" width="680">
</p>

<br/>

Early layers set up character-specific attention patterns using Q-vectors. `John` emerges as the primary belief holder, while `Mark` serves as the secondary actor. The Q-vector for the `box` token (index 57, Mark’s perspective) strongly attends to the `basket` token (index 18, initial room state). K-vectors encode locations—where the `basket` represents the original state and `box` the future location. V-vectors pass initial state information, flowing from the `basket` (index 18) to the `basket` again (index 29), indicating John’s perspective moving forward in the sequence.

O-dimensions reinforce this distinction: `Dim 103: 0.5107` encodes `John` as the main subject, while `Dim 206: -0.1298` keeps character representations separate. Duplicate tokens accumulate in the residual stream, forming distinct initial state representations for both characters.

By the middle layers, indexing sharpens. The `the` token at index 105 acts as a query, attending to noun-object tokens across all three positions with different attention strengths, maintaining context across multiple sequence positions. `box`-related values are passed forward with negative modulation, sustaining temporal context. Detailed parallel state maintenance can be seen in the middle layers across layer 12 in heads 1, 2 and 3.

Q-vectors maintain belief states in 12.1 which tracks subject presence for `John: -0.3388` during departure and `Mark: -0.2272` during action. K-vectors again index locations for `room: 0.3227`: in scene context and `basket: 0.2996`: representing the original location. V-vectors maintain dual states where `Dim 129: 0.0368` encodes `looks` actions and `Dim 43: 0.0317` tracks the `cat` position. O-dimensions then separate perspectives where `Dim 127: 0.2166` maintains room context and `Dim 240: -0.0376` suppresses state updates. 

In 12.2 more actions are integrated, Q-vectors attend to action changes for `Mark: -0.2837` during movement and `puts: 0.3218` for action tracking. K-vectors track locations where `box: -0.1342`: is the new location and `basket: 0.2609`: is the original location. V-vectors encode object movement sequences where `Dim 195: 0.0718` is for transitions and `Dim 123: 0.0337` for object movement.

And in 12.3 we see belief-action integration where Q-vectors maintain temporal context, K-vectors strongly differentiate locations between `box` in the current state and `basket` in the believed state, the V-vectors encode dual perspectives where `Dim 68: 0.0468`: integrates the timing of actions, `Dim 152: 0.0394`: maintains the beliefs, and O-dimensions manage information flow through `Dim 38 0.1374`: maintaining room context and `Dim 219: -0.0996`: encoding temporal boundaries. The fact that this is coming from Mark's perspective is not entirely surprising but its definitely interesting.

The induction circuit activates in layers 2 and 18, linking previous-token heads through K-composition. Attention to the offset diagonal reveals induction peaks at 18.6, and 22.4. Activation patching shows inhibition heads suppressing unwanted token positions, ensuring focus on relevant context. Late-layer induction highlights the basket (index 29) while suppressing the box (index 57), steering final predictions. This reflects strong belief-tracking behavior through the specialized heads.

Shown from strong negative modulations in activation patching, the inhibition heads receive information from duplicate token heads and actively suppress unwanted positions, working with induction heads to guide attention. Also in the late layers, there is an induction path where `the` very strongly attends to `basket` at index 29, and `box` from index 57 is hit with strong negative modulation at the last state of processing before the final prediction.

We can now see the full circuit culminate through multiple specialized heads. In 22.4, Q-vectors show strongest belief encoding where it takes `John: 0.3075`: as the primary perspective and the returning context of `room: -0.1961`. K-vectors maintain temporal boundaries showing `leaves: -0.1719`: as the departure marker and `comes: 0.2158`: as the return marker. V-vectors preserve belief states where `Dim 192: 0.2942`: is the subject encoding and `Dim 142: 0.0846`: provides temporal context. And the O-dimensions show critical integration of `Dim 192: 0.2942`: which strengthens John's perspective and `Dim 155: -0.4181`: suppresses updates during his absence. In 22.2 the Q-vectors track scene changes, the K-vectors index both location states and the V-vectors maintain their separation. 

By 22.3, the circuit achieves false-belief reasoning through careful orchestration where context from John and Mark's perspectives are aggregated at the location state which then maintains distinct state representations. The Q-vectors query temporal context from `away: -0.2288`: representing John's absence period and his return marker with `comes: -0.2810`. K-vectors index locations of the original state `basket: 0.3385` and current state `box: -0.2478`. V-vectors encode dual states of the original location `Dim 151: 0.0552` and current location `Dim 21: 0.0546`.

16.2 then actively prevents belief contamination where Q-vectors track temporal boundaries, V-vectors show strategic inhibition at `basket` (-0.2848): preserving the original belief and `room` (0.4802): maintains context and O-dimensions manage information flow. At 23.5 V-vectors encode final state representation where `Dim 115: 0.1799`: handles the subject encoding, `Dim 233: 0.1708`: maintains room context, and O-dimensions make final integration for `Dim 120: -0.3636`: showing strong suppression of updates and `Dim 237: -0.1991`: enables action inhibition.
