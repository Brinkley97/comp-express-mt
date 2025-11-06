# Three Prompting Strategies for Experimental Setup

## Experiment A: Direct LLM Selection (Control)

### Zero-Shot Prompt
```
You are translating from Akan to English. Select the most appropriate English translation from the options provided.

Akan sentence: "{akan_sentence}"

Translation options:
1. {option_1}
2. {option_2}
3. {option_3}
...

Select the best translation by number only. Respond with just the number (1, 2, 3, etc.).
```

### Few-Shot Prompt
```
You are translating from Akan to English. Select the most appropriate English translation from the options provided.

Examples:
Akan: "Ɔyɛ me maame"
Options: 1. He is my mother 2. She is my mother 3. They are my mother
Selection: 2

Akan: "Mema wo akwaaba"
Options: 1. I welcome you (singular) 2. We welcome you (plural) 3. I welcomed you
Selection: 1

Now select for this sentence:
Akan sentence: "{akan_sentence}"

Translation options:
1. {option_1}
2. {option_2}
3. {option_3}
...

Select the best translation by number only. Respond with just the number (1, 2, 3, etc.).
```

### Chain-of-Thought Prompt
```
You are translating from Akan to English. Follow these reasoning steps to select the most appropriate translation:

Akan sentence: "{akan_sentence}"

Translation options:
1. {option_1}
2. {option_2}
3. {option_3}
...

Step 1: Analyze the Akan sentence structure and identify key linguistic features.
Step 2: Consider what each translation option implies about the context.
Step 3: Determine which option best matches the likely intended meaning.
Step 4: Select the best translation by number.

Provide your reasoning for steps 1-3, then state your final selection as "SELECTION: [number]"
```

---

## Experiment B: LLM Context Generation + Selection

### Zero-Shot Prompt
```
You are analyzing an Akan sentence to infer pragmatic context and select the appropriate English translation.

Akan sentence: "{akan_sentence}"

Translation options:
1. {option_1}
2. {option_2}
3. {option_3}
...

First, infer the pragmatic context by selecting ONE value for each dimension:
- Audience: [Individual | Small_Group | Large_Group | Broadcast]
- Status: [Equal | Superior | Subordinate]
- Age: [Peer | Elder | Younger]
- Formality: [Formal | Casual]
- Gender: [Masculine | Feminine | Neutral]
- Animacy: [Animate | Inanimate]
- Speech_Act: [Question | Answer | Statement | Command | Request | Greeting]

Then, based on these inferred tags, select the most appropriate translation by number.

Respond in this format:
TAGS: Audience=X, Status=X, Age=X, Formality=X, Gender=X, Animacy=X, Speech_Act=X
SELECTION: [number]
```

### Few-Shot Prompt
```
You are analyzing Akan sentences to infer pragmatic context and select appropriate English translations.

Example 1:
Akan: "Ɔyɛ me mpena"
Options: 1. He is my boyfriend 2. She is my girlfriend 3. They are my lover
Analysis: The term "mpena" suggests romantic relationship (casual register), "Ɔ" is 3rd person singular
TAGS: Audience=Individual, Status=Equal, Age=Peer, Formality=Casual, Gender=Masculine, Animacy=Animate, Speech_Act=Statement
SELECTION: 1

Example 2:
Akan: "Yɛfrɛ wo sɛn?"
Options: 1. What is your name? 2. What do they call you? 3. How do we call you?
Analysis: Direct question format, likely addressing individual, neutral formality
TAGS: Audience=Individual, Status=Equal, Age=Peer, Formality=Casual, Gender=Neutral, Animacy=Animate, Speech_Act=Question
SELECTION: 1

Now analyze this sentence:
Akan sentence: "{akan_sentence}"

Translation options:
1. {option_1}
2. {option_2}
3. {option_3}
...

First infer the pragmatic context, then select the best translation.

Respond in this format:
TAGS: Audience=X, Status=X, Age=X, Formality=X, Gender=X, Animacy=X, Speech_Act=X
SELECTION: [number]
```

### Chain-of-Thought Prompt
```
You are analyzing an Akan sentence to infer pragmatic context and select the appropriate English translation. Follow this reasoning process:

Akan sentence: "{akan_sentence}"

Translation options:
1. {option_1}
2. {option_2}
3. {option_3}
...

Step 1: LINGUISTIC ANALYSIS
Examine the Akan sentence for:
- Pronouns and their referents
- Verb forms and tense markers
- Relationship terms or formality markers
- Cultural or contextual clues

Step 2: PRAGMATIC INFERENCE
Based on linguistic features, infer:
- Audience: Who is being addressed? [Individual | Small_Group | Large_Group | Broadcast]
- Status: What is the social relationship? [Equal | Superior | Subordinate]
- Age: What age relationship is implied? [Peer | Elder | Younger]
- Formality: What register is used? [Formal | Casual]
- Gender: What gender is referenced? [Masculine | Feminine | Neutral]
- Animacy: What is being referred to? [Animate | Inanimate]
- Speech_Act: What action is performed? [Question | Answer | Statement | Command | Request | Greeting]

Step 3: TRANSLATION EVALUATION
For each translation option, assess compatibility with inferred pragmatic context.

Step 4: FINAL SELECTION
Choose the translation that best matches the pragmatic profile.

Provide your reasoning for each step, then respond in this format:
TAGS: Audience=X, Status=X, Age=X, Formality=X, Gender=X, Animacy=X, Speech_Act=X
SELECTION: [number]
```

---

## Experiment C: Human-Annotated Tags + LLM Selection

### Zero-Shot Prompt
```
You are selecting the most appropriate English translation for an Akan sentence, given expert-annotated pragmatic context.

Akan sentence: "{akan_sentence}"

Expert pragmatic tags:
- Audience: {audience_tag}
- Status: {status_tag}
- Age: {age_tag}
- Formality: {formality_tag}
- Gender: {gender_tag}
- Animacy: {animacy_tag}
- Speech_Act: {speech_act_tag}

Translation options:
1. {option_1}
2. {option_2}
3. {option_3}
...

Based on the provided pragmatic context, select the most appropriate translation by number only.

Respond with just the number (1, 2, 3, etc.).
```

### Few-Shot Prompt
```
You are selecting appropriate English translations for Akan sentences using expert-annotated pragmatic context.

Example 1:
Akan: "Ɔyɛ me mpena"
Tags: Audience=Individual, Status=Equal, Age=Peer, Formality=Casual, Gender=Masculine, Animacy=Animate, Speech_Act=Statement
Options: 1. He is my boyfriend 2. She is my girlfriend 3. They are my lover
Selection: 1 (masculine gender tag indicates male referent, casual formality fits "boyfriend")

Example 2:
Akan: "Mepɛ sɛ mehunu wo"
Tags: Audience=Individual, Status=Superior, Age=Elder, Formality=Formal, Gender=Neutral, Animacy=Animate, Speech_Act=Request
Options: 1. I want to see you 2. I wish to meet you 3. I'd like to visit you
Selection: 2 (formal register with superior status requires polite phrasing "wish to meet")

Now select for this sentence:
Akan sentence: "{akan_sentence}"

Expert pragmatic tags:
- Audience: {audience_tag}
- Status: {status_tag}
- Age: {age_tag}
- Formality: {formality_tag}
- Gender: {gender_tag}
- Animacy: {animacy_tag}
- Speech_Act: {speech_act_tag}

Translation options:
1. {option_1}
2. {option_2}
3. {option_3}
...

Select the best translation by number only. Respond with just the number (1, 2, 3, etc.).
```

### Chain-of-Thought Prompt
```
You are selecting the most appropriate English translation for an Akan sentence using expert-annotated pragmatic context. Follow this reasoning process:

Akan sentence: "{akan_sentence}"

Expert pragmatic tags:
- Audience: {audience_tag}
- Status: {status_tag}
- Age: {age_tag}
- Formality: {formality_tag}
- Gender: {gender_tag}
- Animacy: {animacy_tag}
- Speech_Act: {speech_act_tag}

Translation options:
1. {option_1}
2. {option_2}
3. {option_3}
...

Step 1: INTERPRET PRAGMATIC TAGS
Explain what each tag implies about the communicative context:
- What does the audience type suggest about the translation?
- How do status and age affect appropriate phrasing?
- What formality level is required?
- What gender/animacy constraints apply?
- What speech act function must be preserved?

Step 2: EVALUATE EACH OPTION
For each translation option, assess:
- Does it match the gender/animacy specifications?
- Does it reflect the appropriate formality level?
- Does it suit the audience and status relationship?
- Does it preserve the speech act function?

Step 3: MAKE SELECTION
Choose the option that best aligns with ALL pragmatic constraints.

Provide your reasoning for steps 1-2, then state your final selection as "SELECTION: [number]"
```

---

## Implementation Notes

### Prompt Template Variables
```python
# For all experiments
akan_sentence: str  # Source Akan sentence
option_1, option_2, ..., option_n: str  # English translation candidates

# For Experiment C only (human-annotated tags)
audience_tag: str  # e.g., "Individual"
status_tag: str  # e.g., "Superior"
age_tag: str  # e.g., "Elder"
formality_tag: str  # e.g., "Formal"
gender_tag: str  # e.g., "Masculine"
animacy_tag: str  # e.g., "Animate"
speech_act_tag: str  # e.g., "Statement"
```

### Expected Output Parsing

**Experiments A and C (Selection Only):**
```python
# Parse output to extract selected number
selection = extract_number(llm_response)
```

**Experiment B (Tags + Selection):**
```python
# Parse structured output
tags = extract_tags(llm_response)  # Dict of tag dimensions
selection = extract_selection(llm_response)  # Selected number
```

### Evaluation Strategy
- **Zero-shot:** Tests pure LLM knowledge and reasoning
- **Few-shot:** Tests learning from minimal examples
- **Chain-of-thought:** Tests explicit reasoning capability

Compare performance across all 9 conditions (3 experiments × 3 prompting strategies) to determine optimal approach for pragmatic context modeling in low-resource MT.