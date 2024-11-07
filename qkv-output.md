# Gemma 2-2B QKV Attention Mechanism Output

<br>

Layer 0, Head 0:
Q head 0 → K head 0 → V head 0:
Q (0.0217) and K (0.3150) attends to: conjunctions and discourse markers
1. when (0.0241)
2. While (0.0240)
3. `<bos>` (0.0240)
4. happened (0.0228)
5. thinks (0.0203)
6. and (0.0195)
7. and (0.0195)
8. and (0.0195)
9. and (0.0195)
10. and (0.0195)
    
V0 outputs match this with high activations for:
1. when (0.2626)
2. While (0.2623)
3. thinks (0.2260)
4. and (0.2189)
5. and (0.2189)
6. and (0.2189)
7. and (0.2189)
8. and (0.2189)
9. and (0.2189)
10. happened (0.2166)
Logit difference after zeroing value vector: 0.9611

- Q/K select connecting words, setting up discourse relations between clauses.
  - Attention Pattern: Attends to function words like `when`, `while`, and `happened`.
  - These layers likely capture structural aspects of sentences and help set up temporal relations and linking between ideas.

<br>

Layer 0, Head 2:
Q head 2 → K head 1 → V head 1:
Q (0.0050) and K (-0.0252) attends to: determiners and punctuation marks
1. `<bos>` (0.2310)
2. the (0.0170)
3. the (0.0170)
4. the (0.0170)
5. the (0.0170)
6. the (0.0170)
7. the (0.0170)
8. the (0.0170)
9. the (0.0170)
10. the (0.0170)
    
V1 outputs match this with high activations for:
1. `<bos>` (1.3412)
2. , (0.1838)
3. , (0.1838)
4. , (0.1838)
5. , (0.1838)
6. , (0.1838)
7. the (0.1690)
8. the (0.1690)
9. the (0.1690)
10. the (0.1690)
Logit difference after zeroing value vector: -0.0811

- Q/K focuses on common articles and punctuation.
  - Attention Pattern: Attends to tokens like the and `<bos>`.
  - Likely focuses on low-content words like determiners to establish grammatical structure.

<br>

Layer 0, Head 3:
Q head 3 → K head 1 → V head 1:
Q (-0.0025) and K (-0.0252) attends to: nsubj1, nsubj2 named entities (proper nouns)
1. John (0.0320)
2. John (0.0320)
3. John (0.0320)
4. John (0.0320)
5. John (0.0320)
6. John (0.0320)
7. In (0.0295)
8. Mark (0.0276)
9. Mark (0.0276)
10. Mark (0.0276)
    
V1 outputs match this with high activations for:
1. John (0.2678)
2. John (0.2678)
3. John (0.2678)
4. John (0.2678)
5. John (0.2678)
6. John (0.2678)
7. In (0.2615)
8. school (0.2216)
9. school (0.2216)
10. Mark (0.2207)
Logit difference after zeroing value vector: -0.0811

- Q/K focuses on nsubjs, specifically named entities like `John` and `Mark`.
  - Attention Pattern: Attends to proper nouns.
  - This head likely helps identify subjects or important actors in the sentence for understanding the main participants.

<br>

Layer 0, Head 7:
Q head 7 → K head 3 → V head 3:
Q (0.0479) and K (0.1089) attends to: prepositions and auxiliary verbs
1. in (0.1918)
2. from (0.1541)
3. on (0.1290)
4. on (0.1290)
5. on (0.1290)
6. to (0.0653)
7. to (0.0653)
8. are (0.0300)
9. In (0.0145)
10. is (0.0142)
    
V3 outputs match this with high activations for:
1. in (2.3593)
2. from (1.8190)
3. on (1.6384)
4. on (1.6384)
5. on (1.6384)
6. to (0.7796)
7. to (0.7796)
8. are (0.3622)
9. In (0.1756)
10. is (0.1713)
Logit difference after zeroing value vector: -0.2213

- Q/K selects positional words like `in`, `from`, and `on`, which establish spatial or temporal relationships.
  - Attention Pattern: Attends to prepositions and auxiliaries.
  - Likely responsible for establishing relationships between objects in a sentence.
  - Likely a copy head:
    - Strong query spikes at regular intervals, minimal key/value interference.
    - Increasing magnitude of Query spikes suggests systematic copying with position awareness.

<br>

Layer 2, Head 5:
Q head 5 → K head 2 → V head 2:
Q (-0.0009) and K (-0.6161) attends to: prepositions and functional words
1. around (0.0767)
2. off (0.0319)
3. on (0.0288)
4. the (0.0268)
5. the (0.0266)
6. the (0.0266)
7. the (0.0256)
8. the (0.0254)
9. the (0.0252)
10. enters (0.0250)
    
V2 outputs match this with high activations for:
1. around (1.0448)
2. off (0.4419)
3. on (0.3096)
4. school (0.2686)
5. enters (0.2657)
6. on (0.2576)
7. on (0.2490)
8. the (0.2375)
9. the (0.2332)
10. the (0.2258)
Logit difference after zeroing value vector: -0.8166

- Q/K focuses on functional words and some spatial markers like `around`, `off`, and `on`.
  - Attention Pattern: Attends to function words and prepositions.
  - Likely helps contextualize spatial relationships or actions involving objects.

<br>

Layer 3, Head 2:
Q head 2 → K head 1 → V head 1:
Q (-0.0063) and K (-0.0055) attends to: determiners and punctuation marks
1. `<bos>` (0.1941)
2. the (0.1732)
3. the (0.1493)
4. the (0.0795)
5. the (0.0767)
6. the (0.0391)
7. the (0.0372)
8. , (0.0173)
9. the (0.0168)
10. , (0.0165)
    
V1 outputs match this with high activations for:
1. the (2.0609)
2. the (1.7489)
3. the (0.9019)
4. the (0.8922)
5. `<bos>` (0.7138)
6. the (0.4756)
7. the (0.4041)
8. the (0.2011)
9. the (0.1864)
10. , (0.1844)
Logit difference after zeroing value vector: 0.0941

- Q/K focuses on articles and punctuation, which serve to maintain sentence structure.
  - Attention Pattern: Attends to determiners like the and punctuation like commas.
  - Likely contributes to parsing sentence structure rather than understanding content.

<br>

Layer 5, Head 0:
Q head 0 → K head 0 → V head 0:
Q (0.0774) and K (0.5294) attends to: determiners, punctuation, nobj3 and locations
1. from (0.0241)
2. . (0.0173)
3. and (0.0172)
4. , (0.0170)
5. school (0.0168)
6. . (0.0168)
7. cat (0.0163)
8. around (0.0159)
9. t (0.0153)
10. cat (0.0151)
    
V0 outputs match this with high activations for:
1. from (0.2719)
2. and (0.1755)
3. . (0.1653)
4. . (0.1629)
5. , (0.1589)
6. around (0.1573)
7. cat (0.1543)
8. cat (0.1474)
9. , (0.1407)
10. and (0.1378)
Logit difference after zeroing value vector: 0.8698

- Q/K identifies tokens like `from`, `cat`, and punctuation.
  - Attention Pattern: Attends to prepositions and nouns.
  - This head seems to balance tracking subjects while noting connecting words.

<br>

Layer 5, Head 2:
Q head 2 → K head 1 → V head 1:
Q (-0.0023) and K (0.0280) attends to: beginnings of sequences and nobj1, nobj2
1. `<bos>` (0.5646)
2. . (0.0414)
3. . (0.0350)
4. box (0.0269)
5. basket (0.0199)
6. . (0.0170)
7. basket (0.0151)
8. basket (0.0135)
9. on (0.0114)
10. . (0.0106)
    
V1 outputs match this with high activations for:
1. `<bos>` (2.5019)
2. . (0.4027)
3. . (0.3524)
4. box (0.2815)
5. basket (0.2080)
6. basket (0.1609)
7. . (0.1600)
8. basket (0.1465)
9. on (0.1214)
10. cat (0.1134)
Logit difference after zeroing value vector: 0.5110

- Q/K attends to early parts of the sequence (`<bos>`, punctuation, and nouns).
  - Attention Pattern: Attends to beginning of sentences and object references like `box` and `basket`.
  - Likely lays the groundwork for object identification in the sentence.

<br>

Layer 6, Head 1:
Q head 1 → K head 0 → V head 0:
Q (0.0106) and K (-0.2177) attends to: nobj1, nobj2, nobj3, dobj1 and determiners
1. `<bos>` (0.2298)
2. box (0.2063)
3. In (0.1397)
4. basket (0.0733)
5. basket (0.0725)
6. . (0.0391)
7. cat (0.0146)
8. room (0.0137)
9. cat (0.0131)
10. . (0.0123)
    
V0 outputs match this with high activations for:
1. box (2.4987)
2. In (2.3379)
3. `<bos>` (1.2710)
4. basket (0.7902)
5. basket (0.7409)
6. . (0.6621)
7. . (0.1808)
8. room (0.1802)
9. cat (0.1627)
10. cat (0.1295)
Logit difference after zeroing value vector: 0.0787

- Q/K identifies sentence parts like `box`, `In`, and nobj3 `cat`.
  - Attention Pattern: Attends to key objects and positional elements.
  - Likely establishes references to specific objects or locations.

<br>

Layer 6, Head 2:
Q head 2 → K head 1 → V head 1:
Q (0.0411) and K (0.2285) attends to: nobj1, nobj2, nobj3 and functional words
1. `<bos>` (0.3725)
2. basket (0.2704)
3. box (0.1783)
4. basket (0.0805)
5. In (0.0184)
6. cat (0.0102)
7. cat (0.0101)
8. the (0.0091)
9. basket (0.0069)
10. ' (0.0049)
    
V1 outputs match this with high activations for:
1. basket (3.2326)
2. box (2.3006)
3. `<bos>` (1.8590)
4. basket (0.9288)
5. In (0.2342)
6. the (0.1255)
7. cat (0.1147)
8. cat (0.1039)
9. basket (0.0831)
10. the (0.0382)
Logit difference after zeroing value vector: 0.9094

- Q/K focuses on the same tokens related to objects, with a slight focus on `basket`, `box`, and determiners.
  - Attention Pattern: Tracks objects and their positions.
  - Likely refines object references, connecting them to the broader sentence structure.

<br>

Layer 6, Head 3:
Q head 3 → K head 1 → V head 1:
Q (0.0621) and K (0.2285) attends to: nobj1, nobj2, nobj3, beginning of sequences and proper nouns
1. `<bos>` (0.7912)
2. basket (0.1111)
3. box (0.0463)
4. basket (0.0297)
5. In (0.0083)
6. the (0.0028)
7. cat (0.0024)
8. cat (0.0015)
9. ' (0.0012)
10. He (0.0011)
    
V1 outputs match this with high activations for:
1. `<bos>` (3.9480)
2. basket (1.3284)
3. box (0.5968)
4. basket (0.3429)
5. In (0.1052)
6. the (0.0384)
7. cat (0.0274)
8. cat (0.0157)
9. He (0.0114)
10. basket (0.0099)
Logit difference after zeroing value vector: 0.9094

- Q/K strongly attends to `<bos>`, `basket`, `box`, and the nobj `cat`.
  - Attention Pattern: Attends to references and key nouns.
  - Likely refines object-to-object, location interactions in the sentence.

<br>

Layer 8, Head 0:
Q head 0 → K head 0 → V head 0:
Q (0.0373) and K (-0.0510) attends to: determiners, beginning of sequences, and nobj1, nobj2, dobj1
1. `<bos>` (0.3343)
2. basket (0.0340)
3. . (0.0275)
4. basket (0.0271)
5. basket (0.0239)
6. box (0.0226)
7. the (0.0215)
8. away (0.0206)
9. the (0.0204)
10. room (0.0180)
    
V0 outputs match this with high activations for:
1. `<bos>` (1.9720)
2. basket (0.5056)
3. basket (0.3914)
4. . (0.3878)
5. box (0.3670)
6. basket (0.3640)
7. away (0.2900)
8. the (0.2847)
9. the (0.2583)
10. room (0.2527)
Logit difference after zeroing value vector: 0.3815

- Q/K focuses on the beginning of the sentence (`<bos>`) and specific object-related words like `basket`.
  - Attention Pattern: Attends to beginnings and references.
  - This head helps anchor the sentence structure around object identification.

<br>

Layer 8, Head 1:
Q head 1 → K head 0 → V head 0:
Q (0.2042) and K (-0.0510) attends to: determiners and beginnings of sentences
1. `<bos>` (0.6773)
2. the (0.0750)
3. the (0.0650)
4. . (0.0242)
5. . (0.0207)
6. , (0.0192)
7. the (0.0167)
8. In (0.0127)
9. on (0.0094)
10. doesn (0.0062)
    
V0 outputs match this with high activations for:
1. `<bos>` (3.9955)
2. the (0.9917)
3. the (0.8217)
4. . (0.3411)
5. . (0.2827)
6. , (0.2607)
7. the (0.2030)
8. In (0.1783)
9. on (0.1289)
10. doesn (0.0802)
Logit difference after zeroing value vector: 0.3815

- Q/K strongly attends to `<bos>`, `the`, and punctuation marks.
  - Attention Pattern: Tracks sequence beginnings and articles.
  - This head contributes to maintaining sentence structure by anchoring positions in sequences.

<br>

Layer 9, Head 5:
Q head 5 → K head 2 → V head 2:
Q (0.0558) and K (-0.3323) attends to: beginning of sequences, nobj1, nobj2 and punctuation
1. `<bos>` (0.2913)
2. box (0.1733)
3. basket (0.0641)
4. . (0.0406)
5. . (0.0404)
6. basket (0.0400)
7. basket (0.0295)
8. , (0.0208)
9. . (0.0162)
10. . (0.0152)
    
V2 outputs match this with high activations for:
1. box (1.6687)
2. `<bos>` (0.8392)
3. basket (0.6620)
4. basket (0.4234)
5. . (0.3894)
6. . (0.3453)
7. basket (0.2621)
8. , (0.1881)
9. . (0.1571)
10. and (0.1361)
Logit difference after zeroing value vector: 0.0839

- Q/K focuses on objects like `box` and `basket`.
  - Attention Pattern: Attends to objects and their relationships.
  - Likely helps with identifying key elements of the sentence by focusing on objects early in the context.

<br>

Layer 10, Head 0:
Q head 0 → K head 0 → V head 0:
Q (0.0706) and K (1.0309) attends to: dobj1, locations, proper nouns and functional words
1. school (0.0370)
2. away (0.0325)
3. know (0.0322)
4. work (0.0310)
5. thinks (0.0276)
6. happened (0.0271)
7. back (0.0232)
8. room (0.0220)
9. school (0.0196)
10. around (0.0183)
    
V0 outputs match this with high activations for:
1. away (0.4870)
2. school (0.4630)
3. work (0.4142)
4. know (0.3939)
5. thinks (0.3556)
6. back (0.3305)
7. room (0.3277)
8. happened (0.3270)
9. from (0.2823)
10. room (0.2799)
Logit difference after zeroing value vector: 0.9721

- Q/K attends to nouns like `school` and `room` while also catching verbs like `thinks` and `happened`.
  - Attention Pattern: Tracks subjects and key actions.
  - Likely important for identifying key entity locations and their actions within the sentence.

<br>

Layer 10, Head 1:
Q head 1 → K head 0 → V head 0:
Q (0.1451) and K (1.0309) attends to: prepositions, articles, and nobj1, nobj2, dobj1
1. `<bos>` (0.0796)
2. on (0.0521)
3. on (0.0477)
4. the (0.0407)
5. box (0.0391)
6. basket (0.0372)
7. basket (0.0329)
8. room (0.0313)
9. . (0.0248)
10. and (0.0222)
    
V0 outputs match this with high activations for:
1. on (0.7748)
2. on (0.7397)
3. the (0.6107)
4. box (0.6087)
5. basket (0.5753)
6. `<bos>` (0.5240)
7. basket (0.5199)
8. room (0.4277)
9. . (0.3573)
10. and (0.3221)
Logit difference after zeroing value vector: 0.9721

- Q/K focuses on function words such as `on`, `the`, and key objects like `box` and `basket`.
  - Attention Pattern: Attends to prepositions and object references.
  - Likely sets up spatial relations between locations and objects in the context of the sentence.
  - Likely a copy head:
    - Strong Query patterns, minimal key interaction.
    - Regular patterns around sentence boundaries suggest systematic copying of structural         
      elements.

<br>

Layer 10, Head 4:
Q head 4 → K head 2 → V head 2:
Q (-0.0295) and K (0.6719) attends to: sentence beginnings and determiners
1. `<bos>` (0.4844)
2. the (0.1378)
3. the (0.0906)
4. box (0.0580)
5. basket (0.0469)
6. basket (0.0306)
7. In (0.0276)
8. . (0.0261)
9. . (0.0205)
10. . (0.0179)
    
V2 outputs match this with high activations for:
1. `<bos>` (3.1027)
2. the (1.5454)
3. the (1.0155)
4. box (0.7331)
5. basket (0.5588)
6. basket (0.3589)
7. . (0.3223)
8. In (0.2727)
9. . (0.2485)
10. . (0.1986)
Logit difference after zeroing value vector: 0.5932

- Q/K focuses on the `<bos>` token and common articles like `the` as well as object markers (`box`, `basket`).
  - Attention Pattern: Attends to function words and object references.
  - Plays a role in grounding the sentence with positional tokens and object identification.

<br>

Layer 10, Head 5:
Q head 5 → K head 2 → V head 2:
Q (0.2333) and K (0.6719) attends to: beginnings of sequences, determiners, nobj1, nobj2
1. `<bos>` (0.7480)
2. the (0.1450)
3. box (0.0181)
4. the (0.0158)
5. the (0.0141)
6. In (0.0116)
7. basket (0.0097)
8. basket (0.0089)
9. the (0.0085)
10. on (0.0065)
    
V2 outputs match this with high activations for:
1. `<bos>` (4.7910)
2. the (1.6268)
3. box (0.2286)
4. the (0.1773)
5. the (0.1479)
6. basket (0.1161)
7. In (0.1140)
8. basket (0.1049)
9. the (0.1032)
10. on (0.0826)
Logit difference after zeroing value vector: 0.5932

- Q/K attends to the start of sentences, articles, and key objects (`box`, `basket`).
  - Attention Pattern: Focuses on sentence structuring elements like `<bos>`, `the`.
  - Likely helps maintain grammatical structure while grounding references to objects.

<br>

Layer 11, Head 4:
Q head 4 → K head 2 → V head 2:
Q (-0.1289) and K (-0.2740) attends to: nobj1, nobj2, nsubj1, verbs and transitions
1. box (0.0601)
2. and (0.0393)
3. basket (0.0384)
4. off (0.0346)
5. takes (0.0340)
6. basket (0.0306)
7. on (0.0282)
8. . (0.0235)
9. and (0.0232)
10. John (0.0228)
    
V2 outputs match this with high activations for:
1. box (0.7796)
2. basket (0.5238)
3. and (0.4166)
4. off (0.3985)
5. basket (0.3786)
6. takes (0.3669)
7. on (0.3343)
8. John (0.2786)
9. on (0.2716)
10. the (0.2348)
Logit difference after zeroing value vector: 0.5752

- Q/K focuses on objects like `box`, `basket`, verbs like `takes` and transitions like `and`, `off`.
  - Attention Pattern: Attends to object manipulation and conjunctions.
  - Likely helps track actions related to objects and subjects and their transitions in context.

<br>

Layer 11, Head 5:
Q head 5 → K head 2 → V head 2:
Q (0.1241) and K (-0.2740) attends to: nobj1, nobj2, nobj3, nsubj2, dobj1, locations and nouns
1. box (0.1410)
2. basket (0.0725)
3. `<bos>` (0.0471)
4. cat (0.0450)
5. cat (0.0428)
6. room (0.0382)
7. cat (0.0352)
8. work (0.0346)
9. school (0.0337)
10. Mark (0.0305)
    
V2 outputs match this with high activations for:
1. box (1.8296)
2. basket (0.9882)
3. cat (0.5773)
4. cat (0.5654)
5. room (0.4690)
6. work (0.4649)
7. cat (0.4255)
8. school (0.4199)
9. John (0.3559)
10. Mark (0.3538)
Logit difference after zeroing value vector: 0.5752

- Q/K attends to key objects (`box`, `basket`, `cat`) and locations (`room`, `school`).
  - Attention Pattern: Focuses on nouns and key objects in the scene.
  - Helps maintain attention on primary subjects and locations, likely for entity tracking.

<br>

Layer 12, Head 0:
Q head 0 → K head 0 → V head 0:
Q (-0.1377) and K (-0.2866) attends to: beginnings of sequences and nobj1, nobj2
1. `<bos>` (0.2039)
2. box (0.1777)
3. basket (0.0517)
4. . (0.0397)
5. . (0.0367)
6. In (0.0276)
7. . (0.0265)
8. basket (0.0254)
9. . (0.0225)
10. happened (0.0174)
    
V0 outputs match this with high activations for:
1. box (2.4872)
2. `<bos>` (1.0572)
3. basket (0.6934)
4. . (0.5244)
5. . (0.4804)
6. In (0.4248)
7. basket (0.3630)
8. . (0.3148)
9. . (0.3067)
10. happened (0.2122)
Logit difference after zeroing value vector: -0.2895

- Q/K focuses on `<bos>`, object markers like `box`, `basket`, and actions (`happened`).
  - Attention Pattern: Tracks objects and beginnings of sentences.
  - Likely helps set up initial references to objects while maintaining sentence structure.

<br>

Layer 12, Head 1:
Q head 1 → K head 0 → V head 0:
Q (0.0589) and K (-0.2866) attends to: sentence beginnings and nobj1, nobj2, dobj1
1. `<bos>` (0.5332)
2. box (0.1805)
3. basket (0.0939)
4. In (0.0402)
5. basket (0.0398)
6. the (0.0212)
7. basket (0.0077)
8. the (0.0060)
9. room (0.0056)
10. on (0.0054)
    
V0 outputs match this with high activations for:
1. `<bos>` (2.7641)
2. box (2.5271)
3. basket (1.2585)
4. In (0.6191)
5. basket (0.5159)
6. the (0.2632)
7. basket (0.1104)
8. the (0.0734)
9. on (0.0719)
10. room (0.0685)
Logit difference after zeroing value vector: -0.2895

- Q/K attends to the `<bos>` token, along with key object markers (`box`, `basket`).
  - Attention Pattern: Focuses on positional and object references.
  - Likely supports maintaining sentence flow and object presence in locations in the scene.

<br>

Layer 12, Head 2:
Q head 2 → K head 1 → V head 1:
Q (-0.1736) and K (0.2436) attends to: nobj1, nobj2, dobj1 and function words
1. box (0.2667)
2. basket (0.1405)
3. `<bos>` (0.0749)
4. basket (0.0713)
5. the (0.0434)
6. on (0.0261)
7. . (0.0236)
8. . (0.0222)
9. room (0.0199)
10. the (0.0165)
    
V1 outputs match this with high activations for:
1. box (4.0378)
2. basket (2.0115)
3. basket (0.9381)
4. the (0.5641)
5. In (0.4808)
6. on (0.3964)
7. `<bos>` (0.3930)
8. . (0.3355)
9. . (0.2892)
10. room (0.2569)
Logit difference after zeroing value vector: 0.7521

- Q/K focuses on key objects (`box`, `basket`) and prepositions (`on`, `the`).
  - Attention Pattern: Attends to objects and their positions in the sentence.
  - Likely tracks object placement and movements.

<br>

Layer 12, Head 3:
Q head 3 → K head 1 → V head 1:
Q (0.5318) and K (0.2436) attends to: nobj1, nobj2 and positional words
1. box (0.4318)
2. basket (0.1340)
3. `<bos>` (0.1219)
4. In (0.0399)
5. the (0.0298)
6. . (0.0252)
7. basket (0.0239)
8. basket (0.0238)
9. . (0.0188)
10. the (0.0157)
    
V1 outputs match this with high activations for:
1. box (6.5370)
2. basket (1.9179)
3. In (1.3363)
4. `<bos>` (0.6396)
5. the (0.3877)
6. basket (0.3426)
7. . (0.3293)
8. basket (0.3147)
9. . (0.2683)
10. the (0.1899)
Logit difference after zeroing value vector: 0.7521

- Q/K attends to object markers (`box`, `basket`) and some positional words (`In`, `the`).
  - Attention Pattern: Focuses on object references and sentence structure.
  - Likely helps ground objects in the context of their positions.

<br>

Layer 12, Head 5:
Q head 5 → K head 2 → V head 2:
Q (-0.0198) and K (1.0940) attends to: nobj1, nobj2, determiners and transitions
1. `<bos>` (0.1973)
2. basket (0.1519)
3. . (0.0662)
4. basket (0.0594)
5. box (0.0584)
6. . (0.0411)
7. In (0.0340)
8. and (0.0320)
9. the (0.0215)
10. the (0.0204)
    
V2 outputs match this with high activations for:
1. basket (1.5999)
2. In (1.0216)
3. box (0.7127)
4. basket (0.7049)
5. `<bos>` (0.6523)
6. . (0.6458)
7. . (0.4664)
8. and (0.3601)
9. the (0.2198)
10. and (0.2159)
Logit difference after zeroing value vector: 1.2017

- Q/K focuses on objects (`basket`, `box`) and transition words (`In`, `and`).
  - Attention Pattern: Tracks objects and their relationships.
  - Likely helps with connecting objects to actions and transitions in the narrative.

<br>

Layer 12, Head 6:
Q head 6 → K head 3 → V head 3:
Q (0.1502) and K (0.4634) attends to: nobj1, nobj2, nobj3, dobj1 and actions
1. box (0.0367)
2. the (0.0349)
3. around (0.0338)
4. the (0.0279)
5. basket (0.0278)
6. on (0.0256)
7. happened (0.0256)
8. basket (0.0248)
9. cat (0.0228)
10. room (0.0222)
    
V3 outputs match this with high activations for:
1. box (0.4652)
2. the (0.3957)
3. around (0.3714)
4. the (0.3544)
5. basket (0.3376)
6. basket (0.3243)
7. on (0.2940)
8. on (0.2840)
9. happened (0.2737)
10. cat (0.2725)
Logit difference after zeroing value vector: 2.4706

- Q/K focuses on objects (`box`, `basket`) and related actions (`around`, `happened`).
  - Attention Pattern: Attends to objects and their associated actions.
  - Likely supports tracking of object manipulation or state changes.

<br>

Layer 14, Head 0:
Q head 0 → K head 0 → V head 0:
Q (0.7226) and K (1.0781) attends to: nobj1, nobj2, nobj3, dobj1, beginning of sequences and named entities
1. basket (0.2567)
2. `<bos>` (0.2215)
3. box (0.1451)
4. cat (0.0826)
5. basket (0.0764)
6. cat (0.0411)
7. basket (0.0275)
8. room (0.0191)
9. room (0.0119)
10. room (0.0119)
    
V0 outputs match this with high activations for:
1. basket (2.7527)
2. box (1.5552)
3. basket (0.8440)
4. cat (0.7962)
5. `<bos>` (0.7445)
6. cat (0.4364)
7. basket (0.3038)
8. room (0.2216)
9. room (0.1273)
10. room (0.1249)
Logit difference after zeroing value vector: 1.3846

- Q/K attends to objects like `basket`, `box`, and named entities (`cat`).
  - Attention Pattern: Focuses on key objects and entities.
  - Likely helps identify the main actors and their relations and locations in the sentence.
  - Likely an induction head:
    - Strong Q/K spike pairs at semantically similar points.
    - Looks for patterns in event sequences.
    - Attention to repeated patterns of actions/states.
    - Patterns suggesting helping to predict next elements in sequences.

<br>

Layer 14, Head 2:
Q head 2 → K head 1 → V head 1:
Q (0.2178) and K (1.0469) attends to: nobj1, nobj2 and functional words
1. box (0.3327)
2. basket (0.1104)
3. basket (0.0382)
4. `<bos>` (0.0376)
5. basket (0.0359)
6. . (0.0313)
7. and (0.0298)
8. off (0.0278)
9. around (0.0249)
10. . (0.0229)
    
V1 outputs match this with high activations for:
1. box (4.7973)
2. basket (1.4255)
3. basket (0.4798)
4. basket (0.4544)
5. . (0.3853)
6. and (0.3602)
7. off (0.3533)
8. around (0.3050)
9. on (0.2578)
10. . (0.2492)
Logit difference after zeroing value vector: 1.3439

- Q/K attends to objects (`box`, `basket`) and function words (`and`, `off`).
  - Attention Pattern: Tracks object-related tokens and transitions.
  - Likely supports maintaining relationships between objects and actions.

<br>

Layer 14, Head 3:
Q head 3 → K head 1 → V head 1:
Q (0.2165) and K (1.0469) attends to: nobj1, nobj2, nobj3 and function words
1. box (0.5125)
2. basket (0.1575)
3. basket (0.1092)
4. cat (0.0168)
5. . (0.0157)
6. basket (0.0146)
7. and (0.0122)
8. the (0.0103)
9. . (0.0098)
10. `<bos>` (0.0084)
    
V1 outputs match this with high activations for:
1. box (7.3884)
2. basket (2.0330)
3. basket (1.3003)
4. cat (0.2197)
5. basket (0.1951)
6. . (0.1930)
7. and (0.1400)
8. the (0.1098)
9. . (0.1064)
10. on (0.0999)
Logit difference after zeroing value vector: 1.3439

- Q/K attends to objects like `box`, `basket`, and some function words (`and`, `the`).
  - Attention Pattern: Focuses on object tracking and sentence structure.
  - Likely maintains references to objects while setting up sentence flow.

<br>

Layer 14, Head 6:
Q head 6 → K head 3 → V head 3:
Q (-0.1490) and K (-0.0532) attends to: nobj1, nobj2, dobj1 and positional words
1. box (0.2976)
2. basket (0.1650)
3. basket (0.1040)
4. `<bos>` (0.0603)
5. the (0.0215)
6. on (0.0197)
7. room (0.0185)
8. the (0.0161)
9. basket (0.0159)
10. off (0.0152)
    
V3 outputs match this with high activations for:
1. box (4.1308)
2. basket (2.4444)
3. basket (1.4092)
4. `<bos>` (0.3255)
5. on (0.2812)
6. room (0.2639)
7. the (0.2600)
8. off (0.2040)
9. cat (0.1991)
10. basket (0.1978)
Logit difference after zeroing value vector: 0.6501

- Q/K attends to objects like `box`, `basket`, and function words like `the`.
  - Attention Pattern: Tracks object-related tokens and some positional markers.
  - Likely helps establish where objects are positioned in the sentence.

<br>

Layer 16, Head 0:
Q head 0 → K head 0 → V head 0:
Q (-0.0574) and K (-0.1461) attends to: spatial references, locations and nobj1
1. `<bos>` (0.1350)
2. room (0.0542)
3. room (0.0503)
4. around (0.0363)
5. room (0.0318)
6. the (0.0309)
7. box (0.0257)
8. the (0.0246)
9. basket (0.0228)
10. , (0.0206)
    
V0 outputs match this with high activations for:
1. room (0.6611)
2. room (0.5263)
3. around (0.4157)
4. room (0.3747)
5. `<bos>` (0.2750)
6. leaves (0.2519)
7. box (0.2441)
8. the (0.2381)
9. room (0.2228)
10. basket (0.2072)
Logit difference after zeroing value vector: -0.0320

- Q/K focuses on spatial terms like `room`, `around`, and objects like `box`.
  - Attention Pattern: Attends to spatial positioning of objects in locations.
  - Likely supports establishing spatial relations of objects in specific locations within the sentence.

<br>

Layer 16, Head 2:
Q head 2 → K head 1 → V head 1:
Q (0.4225) and K (0.8446) attends to: nobj1, nobj2, nobj3, and determiners
1. box (0.1286)
2. basket (0.1185)
3. basket (0.1012)
4. cat (0.0411)
5. cat (0.0298)
6. the (0.0283)
7. cat (0.0252)
8. the (0.0246)
9. it (0.0244)
10. the (0.0189)
    
V1 outputs match this with high activations for:
1. box (2.1364)
2. basket (1.8783)
3. basket (1.6081)
4. cat (0.6608)
5. cat (0.4398)
6. cat (0.4013)
7. it (0.3921)
8. the (0.3451)
9. basket (0.3260)
10. the (0.2663)
Logit difference after zeroing value vector: 0.8602

- Q/K attends to key objects (`box`, `basket`) and named entities like `cat`.
  - Attention Pattern: Tracks objects and actors.
  - Likely helps set up the relationships between entities and objects.

<br>

Layer 16, Head 3:
Q head 3 → K head 1 → V head 1:
Q (0.2005) and K (0.8446) attends to: nobj1, nobj2, nobj3 and named entities
1. cat (0.0956)
2. basket (0.0575)
3. box (0.0550)
4. cat (0.0544)
5. cat (0.0458)
6. it (0.0282)
7. basket (0.0255)
8. on (0.0216)
9. box (0.0215)
10. basket (0.0215)
    
V1 outputs match this with high activations for:
1. cat (1.5355)
2. box (0.9131)
3. basket (0.9114)
4. cat (0.8678)
5. cat (0.6761)
6. basket (0.4670)
7. it (0.4531)
8. box (0.3993)
9. basket (0.3423)
10. on (0.3277)
Logit difference after zeroing value vector: 0.8602

- Q/K focuses on object markers (`box`, `basket`) and named entities (`cat`).
  - Attention Pattern: Tracks objects and entities.
  - Likely maintains attention on important actors and their relations with objects.

<br>

Layer 16, Head 6:
Q head 6 → K head 3 → V head 3:
Q (-0.0548) and K (1.1467) attends to: nobj1, nobj2 and function words
1. basket (0.1278)
2. `<bos>` (0.1263)
3. basket (0.0527)
4. the (0.0491)
5. box (0.0443)
6. the (0.0416)
7. . (0.0346)
8. . (0.0287)
9. the (0.0273)
10. the (0.0265)
    
V3 outputs match this with high activations for:
1. basket (1.5707)
2. basket (0.6091)
3. box (0.5324)
4. `<bos>` (0.5215)
5. the (0.4484)
6. the (0.3474)
7. . (0.3452)
8. In (0.3162)
9. . (0.2937)
10. , (0.2858)
Logit difference after zeroing value vector: 2.5824

- Q/K attends to objects like `basket`, `box`, and function words (`the` `,` `.`).
  - Attention Pattern: Tracks objects and their positioning within the sentence.
  - Likely helps maintain object references and sentence structure.

<br>

Layer 16, Head 7:
Q head 7 → K head 3 → V head 3:
Q (0.1890) and K (1.1467) attends to: nobj1, nobj2 and function words
1. box (0.2632)
2. `<bos>` (0.1595)
3. basket (0.1206)
4. the (0.0977)
5. basket (0.0524)
6. the (0.0459)
7. basket (0.0309)
8. the (0.0299)
9. the (0.0231)
10. the (0.0155)
    
V3 outputs match this with high activations for:
1. box (3.1671)
2. basket (1.3932)
3. the (0.8918)
4. `<bos>` (0.6586)
5. basket (0.6434)
6. basket (0.4012)
7. the (0.3834)
8. the (0.2505)
9. the (0.2133)
10. on (0.1365)
Logit difference after zeroing value vector: 2.5824

- Q/K focuses on objects like `box`, `basket`, and function words (`the`).
  - Attention Pattern: Attends to object references and sentence flow.
  - Likely helps establish relationships between objects and the overall sentence structure.

<br>

Layer 17, Head 0:
Q head 0 → K head 0 → V head 0:
Q (0.3508) and K (0.8263) attends to: nobj1, nobj2, nobj3, dobj1 and locations
1. basket (0.2003)
2. `<bos>` (0.1993)
3. box (0.1580)
4. basket (0.0726)
5. room (0.0326)
6. cat (0.0265)
7. the (0.0201)
8. cat (0.0161)
9. the (0.0160)
10. room (0.0155)
    
V0 outputs match this with high activations for:
1. basket (2.2981)
2. box (1.6071)
3. basket (0.7909)
4. room (0.3902)
5. `<bos>` (0.3314)
6. cat (0.3022)
7. cat (0.1795)
8. room (0.1781)
9. room (0.1627)
10. the (0.1527)
Logit difference after zeroing value vector: 0.6426

- Q/K attends to objects like `basket`, `box`, and locations like `room`.
  - Attention Pattern: Tracks objects and spatial relations.
  - Likely helps maintain attention on key objects and their spatial context in locations.

<br>

Layer 17, Head 3:
Q head 3 → K head 1 → V head 1:
Q (1.1774) and K (1.0032) attends to: nobj1, nobj2 and sentence structure
1. `<bos>` (0.1668)
2. , (0.1120)
3. box (0.1101)
4. basket (0.0852)
5. basket (0.0789)
6. basket (0.0482)
7. box (0.0479)
8. . (0.0445)
9. In (0.0334)
10. the (0.0231)
    
V1 outputs match this with high activations for:
1. box (1.3999)
2. , (1.2220)
3. basket (1.1530)
4. basket (0.9921)
5. In (0.7664)
6. `<bos>` (0.6831)
7. box (0.6386)
8. basket (0.6383)
9. . (0.4601)
10. the (0.2586)
Logit difference after zeroing value vector: 2.1010

- Q/K focuses on objects like `box`, `basket`, and function words (`In`, `the`).
  - Attention Pattern: Tracks objects and sets up sentence structure.
  - Likely helps ground key objects in the overall sentence flow.
  - Likely an induction head:
    - Tracks recurring patterns in nobj states.
    - Predicts likely next states based on previous patterns.
    - Q-V interactions suggest pattern completion behavior.

<br>

Layer 17, Head 4:
Q head 4 → K head 2 → V head 2:
Q (-0.6372) and K (-0.4563) attends to: nobj1, nobj2 and function words
1. `<bos>` (0.1984)
2. basket (0.0994)
3. box (0.0526)
4. the (0.0510)
5. the (0.0402)
6. basket (0.0388)
7. the (0.0373)
8. basket (0.0350)
9. In (0.0312)
10. the (0.0248)
    
V2 outputs match this with high activations for:
1. basket (1.1608)
2. `<bos>` (0.7739)
3. box (0.6273)
4. the (0.5138)
5. basket (0.4946)
6. basket (0.4344)
7. the (0.3857)
8. the (0.3720)
9. In (0.3240)
10. box (0.2930)
Logit difference after zeroing value vector: 0.2356

- Q/K attends to objects like `basket`, `box`, and function words like `the`.
  - Attention Pattern: Tracks objects and sentence structuring elements.
  - Likely helps maintain focus on key objects and their relationships in the sentence. Possibly
    maintaining focus on the context of later mentioned objects, earlier in the sequence.

<br>

Layer 17, Head 6:
Q head 6 → K head 3 → V head 3:
Q (-0.6828) and K (0.3939) attends to: function words and determiners
1. the (0.1646)
2. the (0.1054)
3. the (0.0813)
4. the (0.0747)
5. the (0.0668)
6. `<bos>` (0.0495)
7. the (0.0370)
8. the (0.0367)
9. the (0.0243)
10. the (0.0211)
    
V3 outputs match this with high activations for:
1. the (1.6202)
2. the (1.1319)
3. the (0.8057)
4. the (0.7129)
5. the (0.6911)
6. the (0.3949)
7. the (0.3643)
8. the (0.3315)
9. the (0.2641)
10. `<bos>` (0.2347)
Logit difference after zeroing value vector: 0.0837

- Q/K focuses on function words like `the`, as well as `<bos>`.
  - Attention Pattern: Attends to grammatical structure, especially with determiners.
  - Likely supports sentence cohesion by focusing on low-content words.

<br>

Layer 17, Head 7:
Q head 7 → K head 3 → V head 3:
Q (1.6770) and K (0.3939) attends to: verbs and prepositions
1. off (0.0799)
2. on (0.0717)
3. on (0.0559)
4. is (0.0508)
5. on (0.0416)
6. takes (0.0352)
7. leaves (0.0293)
8. puts (0.0282)
9. it (0.0269)
10. it (0.0223)
    
V3 outputs match this with high activations for:
1. off (1.0384)
2. on (0.7934)
3. is (0.6751)
4. on (0.6458)
5. on (0.4438)
6. takes (0.4101)
7. puts (0.3723)
8. leaves (0.3692)
9. it (0.3462)
10. it (0.2720)
Logit difference after zeroing value vector: 0.0837

- Q/K attends to verbs like `is`, `takes`, and auxiliary words (`it`).
  - Attention Pattern: Focuses on actions and auxiliary verbs.
  - Likely tracks actions related to object manipulation and movement.

<br>

Layer 20, Head 2:
Q head 2 → K head 1 → V head 1:
Q (-0.1035) and K (0.3810) attends to: nobj1, nobj2 and function words
1. basket (0.1123)
2. box (0.0844)
3. basket (0.0803)
4. the (0.0468)
5. the (0.0439)
6. `<bos>` (0.0391)
7. the (0.0319)
8. the (0.0246)
9. the (0.0242)
10. room (0.0237)
    
V1 outputs match this with high activations for: nobj1, nobj2, nobj3 and function words
1. basket (1.1895)
2. box (0.9763)
3. basket (0.6426)
4. the (0.3359)
5. the (0.3163)
6. cat (0.2786)
7. cat (0.2717)
8. the (0.2504)
9. basket (0.2498)
10. room (0.1792)
Logit difference after zeroing value vector: 0.1956

- Q/K attends to objects like `basket`, `box`, and function words (`the`).
  - Attention Pattern: Tracks key objects and sentence structure.
  - Likely helps maintain the presence of key objects in the sentence context.

<br>

Layer 20, Head 3:
Q head 3 → K head 1 → V head 1:
Q (0.5209) and K (0.3810) attends to: nobj1, nobj2, dobj1, locations and spatial references
1. `<bos>` (0.0622)
2. basket (0.0489)
3. box (0.0421)
4. basket (0.0389)
5. room (0.0365)
6. school (0.0360)
7. on (0.0311)
8. basket (0.0305)
9. the (0.0260)
10. on (0.0225)
    
V1 outputs match this with high activations for: nobj1, nobj2, nobj3, dobj1, locations and spatial references
1. basket (0.5405)
2. basket (0.5180)
3. box (0.4871)
4. school (0.4304)
5. room (0.3661)
6. box (0.2813)
7. room (0.2613)
8. basket (0.2442)
9. cat (0.2264)
10. on (0.2052)
Logit difference after zeroing value vector: 0.1956

- Q/K focuses on objects like `basket`, `box`, and locations (`room`, `school`).
  - Attention Pattern: Attends to objects and spatial references.
  - Likely maintains attention on key objects and their spatial positioning in locations.

<br>

Layer 20, Head 7:
Q head 7 → K head 3 → V head 3:
Q (0.7354) and K (0.1601) attends to: nobj1, nobj2, dobj1, locations and sentence beginnings
1. `<bos>` (0.3885)
2. basket (0.1074)
3. basket (0.1029)
4. box (0.0558)
5. basket (0.0502)
6. box (0.0296)
7. school (0.0200)
8. room (0.0158)
9. room (0.0145)
10. the (0.0110)
    
V3 outputs match this with high activations for: nobj1, nobj2, nobj3, dobj1, and sentence beginnings
1. basket (1.5529)
2. `<bos>` (1.4056)
3. basket (1.3173)
4. box (0.7195)
5. basket (0.5218)
6. box (0.5130)
7. school (0.2191)
8. room (0.2060)
9. room (0.1861)
10. cat (0.1154)
Logit difference after zeroing value vector: -0.1561

- Q/K attends to objects like `basket`, `box`, and sequence markers (`<bos>`).
  - Attention Pattern: Focuses on object references and sentence structuring.
  - Likely helps establish sentence context and object-location relationships.

<br>

Layer 22, Head 2:
Q head 2 → K head 1 → V head 1:
Q (-0.5127) and K (0.8859) attends to: sentence beginnings, nobj1, nobj2, nsubj2 and locations
1. `<bos>` (0.7851)
2. basket (0.0310)
3. basket (0.0217)
4. basket (0.0115)
5. work (0.0091)
6. the (0.0065)
7. Mark (0.0065)
8. school (0.0053)
9. puts (0.0050)
10. the (0.0044)
    
V1 outputs match this with high activations for:
1. `<bos>` (0.6593)
2. basket (0.3364)
3. basket (0.2063)
4. work (0.1032)
5. Mark (0.0920)
6. basket (0.0864)
7. takes (0.0531)
8. school (0.0465)
9. puts (0.0448)
10. box (0.0436)
Logit difference after zeroing value vector: 0.5646

- Q/K focuses on the beginning of the sentence (`<bos>`) and object locations like `basket`, `work`.
  - Attention Pattern: Attends to sentence structuring elements and key objects.
  - Likely helps maintain references to objects and entities early in the sentence.

<br>

Layer 22, Head 4:
Q head 4 → K head 2 → V head 2:
Q (6.9521) and K (6.8051) attends to: nobj1, nobj2, nobj3, nsubj1, dobj1 and sentence beginnings
1. `<bos>` (0.3358)
2. basket (0.1275)
3. basket (0.1132)
4. box (0.0788)
5. basket (0.0700)
6. box (0.0525)
7. cat (0.0448)
8. cat (0.0235)
9. room (0.0181)
10. John (0.0157)
    
V2 outputs match this with high activations for:
1. basket (1.8161)
2. basket (1.3706)
3. box (0.9736)
4. box (0.8343)
5. basket (0.6768)
6. `<bos>` (0.6349)
7. cat (0.5951)
8. John (0.3181)
9. cat (0.2764)
10. room (0.2628)
Logit difference after zeroing value vector: 7.2532

- Q/K attends to objects like basket, box, and the beginning of the sentence (`<bos>`).
  - Attention Pattern: Focuses on object identification and sentence flow.
  - Likely helps establish relationships between locations, subjects, objects and other sentence elements from earlier context, showing importance of beginning of sequence context for nsubj1's belief state.

<br>

Layer 22, Head 5:
Q head 5 → K head 2 → V head 2:
Q (0.5277) and K (6.8051) attends to: nobj1, nobj2, nobj3, dobj1, locations and sentence beginnings
1. basket (0.2080)
2. `<bos>` (0.1900)
3. basket (0.1416)
4. box (0.0984)
5. box (0.0681)
6. basket (0.0526)
7. cat (0.0312)
8. cat (0.0183)
9. school (0.0177)
10. room (0.0165)
    
V2 outputs match this with high activations for: nobj1, nobj2, nobj3, dobj1, nsubj1 and sentence beginnings
1. basket (2.5193)
2. basket (1.3700)
3. box (1.2153)
4. box (1.0817)
5. basket (0.7484)
6. cat (0.4146)
7. `<bos>` (0.3592)
8. John (0.2432)
9. room (0.2398)
10. cat (0.2150)
Logit difference after zeroing value vector: 7.2532

- Q/K focuses on objects like `basket`, `box`, and the start of the sentence (`<bos>`).
  - Attention Pattern: Attends to objects and sentence flow.
  - Likely supports grounding of objects within the sentence structure.

<br>

Layer 23, Head 5:
Q head 5 → K head 2 → V head 2:
Q (-2.2033) and K (-2.0187) attends to: nobj1, nobj2, nobj3, dobj1 and named entities
1. basket (0.3339)
2. basket (0.1885)
3. basket (0.1350)
4. box (0.0457)
5. cat (0.0344)
6. cat (0.0260)
7. cat (0.0198)
8. `<bos>` (0.0195)
9. box (0.0193)
10. room (0.0105)
    
V2 outputs match this with high activations for:
1. basket (3.1609)
2. basket (1.5867)
3. basket (1.4373)
4. box (0.4255)
5. cat (0.3514)
6. box (0.2541)
7. cat (0.2039)
8. cat (0.2021)
9. room (0.0813)
10. room (0.0761)
Logit difference after zeroing value vector: -2.0844

- Q/K attends to objects like `basket`, `box`, and named entities like `cat`.
  - Attention Pattern: Focuses on key objects and entities.
  - Likely helps maintain attention on primary actors and their relation to objects and locations.

<br>

Layer 23, Head 6:
Q head 6 → K head 3 → V head 3:
Q (0.8550) and K (0.9377) attends to: nobj1, nobj2, nobj3, dobj1 and sentence beginnings
1. `<bos>` (0.3566)
2. basket (0.1319)
3. basket (0.1096)
4. box (0.0676)
5. cat (0.0666)
6. basket (0.0454)
7. box (0.0446)
8. cat (0.0245)
9. room (0.0160)
10. room (0.0137)
    
V3 outputs match this with high activations for:
1. basket (1.8120)
2. basket (1.2400)
3. cat (0.8907)
4. box (0.7724)
5. box (0.7115)
6. `<bos>` (0.6792)
7. basket (0.4075)
8. cat (0.2867)
9. room (0.1956)
10. room (0.1712)
Logit difference after zeroing value vector: 1.6867

- Q/K focuses on objects like `basket`, `box`, and the start of the sentence (`<bos>`).
  - Attention Pattern: Tracks object-related tokens and sentence flow.
  - Likely helps establish sentence structure and maintain object references in relation to locations.

<br>

Layer 24, Head 3:
Q head 3 → K head 1 → V head 1:
Q (0.2262) and K (0.0698) attends to: function words and prepositions
1. on (0.0846)
2. on (0.0775)
3. `<bos>` (0.0671)
4. the (0.0519)
5. off (0.0417)
6. the (0.0358)
7. the (0.0320)
8. the (0.0295)
9. on (0.0292)
10. the (0.0270)
    
V1 outputs match this with high activations for:
1. on (0.7154)
2. on (0.6642)
3. the (0.4357)
4. off (0.3155)
5. the (0.2894)
6. the (0.2774)
7. the (0.2551)
8. on (0.2318)
9. the (0.2247)
10. the (0.1668)
Logit difference after zeroing value vector: 0.0279

- Q/K focuses on function words (`the`, `on`, `off`) and prepositions.
  - Attention Pattern: Attends to sentence structuring elements.
  - Likely supports grammatical relations between objects and actions.

<br>

Layer 25, Head 2:
Q head 2 → K head 1 → V head 1:
Q (-0.0176) and K (-0.1950) attends to: sentence beginnings, nobj1, nobj3, locations and determiners
1. `<bos>` (0.1735)
2. basket (0.0418)
3. on (0.0273)
4. the (0.0264)
5. on (0.0263)
6. cat (0.0260)
7. on (0.0260)
8. school (0.0243)
9. the (0.0235)
10. the (0.0215)
    
V1 outputs match this with high activations for:
1. basket (0.2819)
2. cat (0.2399)
3. `<bos>` (0.1990)
4. on (0.1742)
5. the (0.1630)
6. on (0.1490)
7. box (0.1488)
8. school (0.1481)
9. on (0.1414)
10. the (0.1375)
Logit difference after zeroing value vector: -0.1202

- Q/K attends to objects (`basket`, `cat`) and determiners (`the`).
  - Attention Pattern: Focuses on objects and sentence structuring elements.
  - Likely helps maintain reference to objects within the sentence context.

<br>

Layer 25, Head 4:
Q head 4 → K head 2 → V head 2:
Q (2.3222) and K (1.8264) attends to: prepositions and determiners
1. on (0.0631)
2. on (0.0553)
3. the (0.0458)
4. the (0.0417)
5. the (0.0401)
6. `<bos>` (0.0388)
7. the (0.0373)
8. off (0.0339)
9. on (0.0326)
10. takes (0.0305)
    
V2 outputs match this with high activations for:
1. on (0.4699)
2. on (0.4105)
3. the (0.3519)
4. the (0.3254)
5. the (0.3140)
6. the (0.3006)
7. the (0.2635)
8. on (0.2413)
9. off (0.2297)
10. the (0.2289)
Logit difference after zeroing value vector: 1.2342

- Q/K focuses on prepositions (`on`, `off`) and determiners (`the`).
  - Attention Pattern: Attends to sentence structuring elements and positional words.
  - Likely helps establish object placements and actions in the sentence.

<br>

Layer 25, Head 5:
Q head 5 → K head 2 → V head 2:
Q (-0.3728) and K (1.8264) attends to: determiners and sentence beginnings
1. `<bos>` (0.2163)
2. the (0.0529)
3. the (0.0436)
4. the (0.0327)
5. the (0.0325)
6. the (0.0264)
7. the (0.0221)
8. the (0.0210)
9. the (0.0134)
10. . (0.0134)
    
V2 outputs match this with high activations for:
1. the (0.4863)
2. the (0.4280)
3. the (0.2650)
4. the (0.2621)
5. `<bos>` (0.2102)
6. the (0.1984)
7. the (0.1713)
8. the (0.1563)
9. the (0.1032)
10. . (0.0918)
Logit difference after zeroing value vector: 1.2342

- Q/K attends to determiners like `the`, and the start of the sentence (`<bos>`).
  - Attention Pattern: Focuses on low-content words like the.
  - Likely helps maintain sentence cohesion and structure.

<br>

Layer 25, Head 7:
Q head 7 → K head 3 → V head 3:
Q (0.6640) and K (0.3512) attends to: determiners and articles
1. the (0.0235)
2. the (0.0230)
3. a (0.0224)
4. the (0.0220)
5. the (0.0220)
6. a (0.0215)
7. the (0.0200)
8. to (0.0186)
9. the (0.0173)
10. the (0.0170)
    
V3 outputs match this with high activations for:
1. the (0.2124)
2. the (0.2102)
3. the (0.2089)
4. the (0.2081)
5. a (0.2022)
6. a (0.2013)
7. to (0.1909)
8. the (0.1853)
9. on (0.1673)
10. from (0.1631)
Logit difference after zeroing value vector: -0.7083

- Q/K focuses on determiners (`the`, `a`) and function words.
  - Attention Pattern: Tracks low-content words for sentence structure.
  - Likely helps maintain grammatical cohesion and object references.
