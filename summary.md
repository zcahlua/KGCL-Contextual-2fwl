# Summary

## ChatGPT Shared Conversation: Dataset Bond Separation Questions

Source: https://chatgpt.com/share/6a394db7-60e4-83ee-924b-a7c1bd88f557

The conversation discusses KGCL-style retrosynthesis modeling on USPTO-50K, focusing on bond centers, atom centers, and sparse 2-FWL reasoning. A key distinction is that a reaction may be a single retrosynthesis step while still requiring multiple graph edits. Cutting one bond only separates atoms when that bond is a graph-theoretic bridge; rings or atoms with multiple incident bonds may require more than one bond edit to fully disconnect a target atom or fragment.

The discussion separates several related concepts:

- A bond center is an existing product bond that may undergo deletion or bond-type modification.
- An atom center is an atom-level reaction site, such as atom modification or leaving-group attachment.
- A 2-FWL intermediate or bridge atom is a message-passing construct between pair states, and should not automatically be treated as only a common chemical neighbor unless the carrier pair set is restricted to actual molecular bonds.

For a bond-focused sparse 2-FWL model, it is reasonable to score only existing bond centers:

```text
S_score = {(i,j), (j,i) : {i,j} in E_t}
```

However, the message-passing carrier set may need to be larger than the scored bond set so that intermediate atoms used by 2-FWL are not artificially limited. Restricting the intermediate set to `N(i) intersect N(j)` is only correct under a strict design where carrier pair states exist only for actual molecular bonds.

The main modeling conclusion is that bond centers can be the primary sparse pair objects, but atom centers should not be removed entirely. KGCL defines atom actions, bond actions, and a stop action. If atom centers are ignored, reactions requiring atom-level edits, such as leaving-group attachment after a bond deletion, cannot be reproduced exactly. This can cause the decoder to stop too early or compensate with incorrect bond edits. A safer architecture is therefore bond-focused scoring with a separate lightweight atom-edit head.
