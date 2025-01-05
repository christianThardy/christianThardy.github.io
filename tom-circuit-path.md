# ToM circuit path

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
   - 16.7 queries the output of 2.3 isolating Mark's state/actions across the entire sequence
     - Keys focus on John's temporal state while he's away and his last phrase (`John thinks the cat is on the`)
       - Very low attention on Mark's state when he moves the cat, but stronger attention on `basket` and `box`
       - Values project high strength on `basket`, `on`, `leaves`, `enters` from 2.3 outputs
         
   - 16.7 queries the keys of 10.5, Focus on determiners/adpositions at the sequence’s beginning and end
     - Keys attend to output and query focusing on auxiliary verbs and temporal markers, with a `John` bias.
       -  Values project function words from the keys, emphasizing John/Mark’s initial states

  -16.7 queries the output of 16.2 emphasizing John being away and then coming back, correlating each instance in the sequence
  - Queries the keys to encode possible cat locations; high attention to John's actions
    -  Values encode Mark’s cat placement with minimal attention to John’s perspective
 
  - 18.6 queries from all processed streams, primarily from induction head outputs
    - 5.4’s keys: Strongly focus on `Mark` moving `the cat`, John’s absence, and the final phrase
    - 18.6 keys encodes 5.4's queries, centering John’s initial state, Mark’s movement, and John returning. Values project this info forward
      - Values project this information forward

    - 18.6 queries the keys of 15.0, emphasizes parallels—`John` leaving for `school` vs. `Mark` leaving for `work`—and when `John` put `the cat` on `the basket`. Values project queries about their comings/goings and the cat’s position
      
    - 22.2 queries 2.5 output, encodes `John`/`Mark` departures, John’s return, with a focus on `He leaves the room`, `doesn’t know what happened` and `what happened` while he was away
      - Keys attend to queries focusing on the initial state of `the room`, temporal markers like `While`, and `what happened in the room while he was away`

    - 22.2 queries 16.2 keys, focusing on  initial state, John’s placement of the `cat`, `Mark` moving it, and the final phrase. Most attention on Mark’s actions
    - 6.2 queries encode John's unawareness, and 22.2 keys attend to pronouns and cat locations
      - 18.7's keys attend to the initial state, `cat` movements, and final phrase, 22.2's queries emphasize Mark’s move
        - 18.7's queries encode `John` returning, 22.2's keys highlight Mark’s move and departure
          - Values project the output of 18.7 in regards to Mark's actions beyond the initial parts of the sequence forward

    - 22.2 queries the output of 21.5, focusing on the initial state of `John`, `Mark` and `the cat`, positions in the sequence where `the cat` was moved, both with strong attention on John's unawareness and the `box`/`basket`'s relation to the final phrase of the sequence
      - Queries focus on keys which attend mainly to `John`
        - Keys focus on `cat` location changes and John’s unawareness
        - Keys attend to the queries with a primary focus on the final phrase in the sequence and the initial state `John`, `Mark`, `the cat` and `the room`
          - Values focus on `on`/`off` relationships for the cat/objects

    - 22.4 queries encode 2.3 keys that attend to all `cat` locations in the sequence, subject movements, and John's unawareness
      - Keys attend to queries, focusing on John's initially placement of `the cat` on `the basket`, `Mark` moving `the cat`, with a strong focus on the keys attending to `Mark` moving `the cat` and the queries encoding John's unawareness
        - Values encoding and projecting the the output where its focused on `John` in the beginning of the sequence, his actions throughout and heavy attention on `John` `thinking`
   
    - 22.4 queries the keys of 8.1, keys heavily attend to the initial mention of `John` and all subsequent mentions with varying attention strength. Queries encode the initial phrase of the sequence
    - 22.4 keys attend to 8.1 queries, focusing on duplicate/similar phrases, with high attention on temporal markers
   
    - 22.4 queries the keys of 10.5, keys heavily attend to `Mark takes the cat off the basket and puts it on the box` and `John looks around the room`. Queries focus on John placing cat on the basket, leaving, and John’s state
      - Keys attend to the output, where most of the focus is on `John` moving the `cat`, and John's unawareness, which is heavily attending to `John takes the cat and puts it on the basket`
        - Values project that information and temporal markers forward

    - 22.4 queries the output of 15.0, both focus on the seperate perspective of `Mark`/`John` and actions toward the `cat`
      - 22.4 queries the keys of 15.0, both focus on the state of `the room` at all positions in the sequence, with a heavy focus on John's unawareness of the `room` while he was away
        - 22.4 keys attend to 15.0 outputs, focusing on `John`/`Mark` leaving the `room`, with most attention on Mark's action, `John` unawareness and John's initial actions
          - Keys attend to queries, focusing on `the cat` at the beginning of the sequence, and from John/Mark's perspective
   
    - 22.4 queries, focusing on `John`/`Mark` moving the `cat` between the `basket`/`box`, query the output of 16.7, which focuses on `John` leaving for `school`, with more focus on `John` returning
      - Queries encode keys about determiners, subjects, objects, and locations, emphasizing actions in the room
   
    - 22.4 queries the output of 18.6, focusing on entire phrases. The output, where `John` initially places the `cat`, highly correlating with his unawareness of how things changed denoted by the queries
      - Keys attending to queries with strong correlation between `Mark` leaving and the room's initial state
   
    - 22.4 queries the output of 18.7, output focuses on John’s absence/return/thoughts on cat’s location. Queries emphasize the initial state, cat movements, and John’s unawareness correlating to the last phrase of the sequence
      - Queries encode the keys, heavily attending to `John`/`Mark` and the initial state of the sequence
        - Keys attend to queries, heavily focused on the initial state of `the room` in relation to `John` looking around `the room` and his unawareness
          - Values heavily encode Mark's actions
   
    - 22.4 queries the output of 20.2 and focuses on Mark's actions relative to John's
      - Keys heavily attend to the `leaves` positions in 20.2's values. The keys also attend to Mark's action of moving the cat in 20.2's output, which is correlating with `John` leaving the `room` and returning from `school`
        - Values encode the basket’s initial position and temporal markers.

    - 22.4 queries the output of 21.5, focusing on the state of the entire sequence as `John` is away at `school`, with most attention on the initial state of `the room` at the beginning of the sequence
      - Keys attend to the output, where the initial state of the `room` is correlating highly with John's actions, ignoring most of Mark's actions. Keys also attend to queries focusing on `John` leaving, comparing that to every phrase in the sequence with the most focus on John's return and unawareness of changes made by `Mark`
        - Values draw from the output and focus heavily on the initial state of the room
   
    - 22.4 queries the keys of 22.3, focusing on the initial state of the `room` and `John` leaving the room with strong attention across `John`/`Mark`'s perspective when they seperately moved the `cat`
      - Keys attend to the output and queries, showing a more focused representation of the Q/K relationship. Keys also attend to queries, showing the same relationship, but with heavy attention on `Mark` leaving, `John` returning and how those perspectives correlate to John's unawareness of the cat's new position

    - 22.5 queries keys of 5.4, focusing on `John comes back`, `knows` and `the cat`, while simultaneously attending to `John`/`Mark` leaving the `room`
      - Keys attending to queries, focusing on the initial state of the `room` and `John` initially moving the `cat`
        - Values are sparse but encode the keys and projects `John` coming back to `the room` and `school` with heavy attention
   
    - 22.5 queries the output of 6.2, encodes token positions related to `John`, `the room`, his actions and temporal markers. Queries then focus on keys and encode token positions related to the initial state of the `room`, `John` being away, `Mark` moving the `cat` then leaving and `John` returning
      - Keys only attend to queries related to `John`/`Mark` leaving, and `John` return
        - Values only project temporal markers from the keys forward

    - 22.5 queries the keys of 14.3, keys attend to `Mark` leaving the `room`, queries encode that, John's unawareness and heavily focuses on instances of `the basket`, `the box`, `the room`, `the cat` across the entire sequence
      - Keys attend to output token positions, correlating `John` initially moving the `cat`, with most of the attention on `Mark` moving the `cat and leaving the `room`

    - 22.5 queries the output of 16.2, focusing on `John` and `Mark` moving the `cat`, and the initial state of the room, with most of the attention on `John`. 16.2's keys attend to John's unawareness while 22.5's queries heavily focus on `Mark` moving the `cat`

    - 22.5 queries the output of 18.7 showing equal strength to the initial state of the `box`/`basket` while encoding the position that marks John's unawareness
      - Keys heavily attend to outputs regarding John's unawareness and his action of looking around `the room`. Keys also attend to the inital state of `the room` from the query positions
        - Values receive 18.7's output and projects heavily activated `Mark` tokens

    - 22.5 queries the keys of 22.3, sparsely focusing on `cat` across the sequence
      - Keys attend to the output, focusing on `John`/`Mark` moving `the cat`. Keys also attend to the query, showing heavy correlation between `John`/`Mark` leaving and John's unawareness
        - Values encodes prior queries forward
   
    - 22.5 queries the output of 22.4, focusing on the last phrase signally John's unawareness and where he `thinks` the `cat` is
      - Queries encode the keys, showing a strong previous token pattern across the entire sequence, with strong attention on the beginning of the sequence and John's awareness of where he put the `cat`. Mark's perspective is sparsely represented
        - Keys attend to the output, keys show the strongest attention on where `John` thinks the `cat` is, `Mark` moving the `cat`, where `John` placed the `cat`, and the outputs strongest attention on `John` leaving, `Mark` moving the `cat` and where `John` thinks it is
   
    - 23.6 queries the keys of 8.1, the keys attend to John's actions/unawareness/where he thinks `the cat` is, and Mark's actions, while the queries heavily encode `John` moving `the cat`/leaving/`Mark` moving `the cat` while focusing on `Mark` moving `the cat`, often focusing on duplicate tokens
      - 23.6 keys attending to queries of 8.1, `John` at the position where he comes back strongly attending to the initial location/where he moved the `cat`/where `Mark` moved the `cat`
        - Values projecting tokens from 8.1's keys associating the initial state of `the room` with `John` returning to the room
   
    - 23.6 queries the keys of 11.3, keys attend to tokens across the sequence, focusing the most on the initial state of the `room`, `Mark` moving the `cat`, and `John` returning to/looking around the `room`. Queries heavily encoding the `cat`, `box`, `basket`, `know` and `thinks` in relation to `John`
   
    - 23.6 queries the keys of 14.2, keys attend to the initial state of the `room`, and `Mark` moving the `cat`. Queries encode `Mark` moving the `cat`, `John` returning to the room and heavily focusing on John's unawareness
      - Keys attend to the outputs, outputs encoding `Mark` moving the `cat`/`John` returning, keys heavily attending to the initial state of the `room`, `John`/`Mark` moving the `cat`, and where `John` starts to `think` where the `cat` may be
        - Keys attend to queries, queries encode everything except `Mark` leaving the `room`, keys heavily attend to `John` being `away`, `Mark` leaving the `room`, `John` coming back and his unawareness
   
    - 23.6 queries the output of 14.3, output encodes transition words, punctuation, and `Mark` moving the `cat`. Queries heavily encode `Mark` moving the `cat` and John's unawareness
      - Queries encode the keys, where keys focus on the initial state of `the room` and `Mark` moving `the cat`. Queries heavily encode John's unawareness
        - 23.6 keys attend to 14.3 queries, queries encode everything except `Mark` leaving `the room` to go to `work`. Keys heavily attend to `John`/`Mark` leaving `the room`, `John` looking around the room and John's unawareness
   
    - 23.6 queries the output of 17.0, output encodes `Mark` moving the `cat`/leaving, `John` returning/his unawareness. Queries heavily encode `Mark` moving the `cat` where `John` comes back/is unaware and where `John` is unaware and when he comes back
      - Queries encode the keys. Keys attend to John's unawareness, queries heavily encode `Mark` moving the `cat`, `John` coming back and John's unawareness
   
    - 23.6 queries the keys of 22.4, keys attend to the entire sequence with a focus on `John` leaving the `room`, `Mark` leaving the `room` and John's unawareness. Queries heavily encode `John`/`Mark` moving the `cat`, `John` coming back, and `John` `thinking` about where the `cat` is
      - Keys attend to the output, output encodes `John` moving the `cat`, `Mark` moving the `cat`. Keys heavily attend to `John` leaving for `school`, `Mark` leaving for `work`, `John` returning and `John` thinking about where the `cat` is
        - Keys attend to queries, queries encode the entire sequence, keys heavily attend to the initial state of the `room`, `John`/`Mark` moving the `cat` and John's unawareness
          - Values project from the outputs, outputs encode the entire sequence, values heavily encode prior query and key positions
   
    - 23.6 queries the output of 22.5, outputs encode the entire sequence, queries heavily encode John's actions, and `John` coming back
      - Queries encode the keys, keys attend to `John`/`Mark leaving, and John's unawareness. Queries heavily encode `Mark` moving the `cat`, `John` looking around the `room` and thinking about where the `cat` is
        - Keys attend to the output, output encodes the initial state of `the room`, `Mark` leaving `the room` and `John` returning. Keys heavily attend to `John` coming back to `the room` and the `cat` at the last position in the sequence
       
**Copy Suppression Processing (L14-23)**
- **Primary Function:** State filtering and arbitration
    - **QKVO Flow:**
      - 14.3 queries 11.3, 12.1, 12.2, 12.3 previous token heads for initial state filtering, focusing on `John` leaving the room/coming back, and `Mark` moving the `cat`
        - Keys attend to outputs of 8.1, outputs focus on the entire sequence except for the initial state. Keys heavily attend to `John`/`Mark` leaving the room, `John` coming back and where he thinks the cat is
          - Values encode 8.1's keys, projecting where `John` moved the `cat` in relation to `John` coming `back` to the `room`, where the `box` is at Mark's position when he moved the cat, forward 
          
      - 16.2 keys attend to the queries of 8.1, queries encoding the initial state of the room, `John`/`Mark` moving the cat, and `the` at the end of the sequence. Keys heavily attending to the same positions in the sequence
        - Values encode 8.1 keys, projecting the initial state of the sequence and `Mark` moving the cat forward
       
      - 16.2 queries the keys of 14.2, encoding `Mark`'s actions and `John`'s unawareness
        - Keys attend to the output encoding the absence of `John` and the departure of `Mark`
          -  Values encode the keys, projecting when `John`/`Mark` `leaves` forward
      
      - 18.7 queries the output of 14.3, encoding John's unawareness, specifically `He` at the position when he leaves for school and `doesn't know what happened in the room` near the end of the sequence

      - 18.7 queries the output of 16.2, heavily encoding `John` being away simultaneously with `John thinks the cat is on the`
        
      - 20.2 queries the keys of 14.2, encoding the room's initial state, `Mark` moving the `cat`, and the sequence's final phrase, with a strong focus on Mark's action and the phrase `is on` at the end of the sequence
        
      - 20.2 keys attend to the output of 16.2, heavily focusing on `Mark` moving the `cat`/leaving the `room`, simultaneuously to `the cat is on the` at the end of the sequence

      - 23.5 query and key the output of the late group of previous token heads to encode `Mark` moving the `cat` to the `box`, and leaving for `work` with the last phrase of the sequence and projects those values forward to the final layers
