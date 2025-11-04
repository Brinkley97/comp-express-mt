from abc import ABC, abstractmethod

class BasePromptFactory(ABC):
    def get_numbered_prompt(self, sentences):
        # Build a numbered list from the translations
        numbered_options = [
            f"\t{i}. {translation}"
            for i, translation in enumerate(sentences)
        ]

        # Join the list into a single block of text, each option on its own line
        options_block = "\n".join(numbered_options)

        return options_block

class ZeroShotPromptFactory(BasePromptFactory):
    def __init__(self, type):
        self.type = type

    def get_name(self):
        if self.type == "direct":
            return "zero_shot-direct"
        elif self.type == "context":
            return "zero_shot-context"

    def get_role_prompt(self):
        if self.type == "direct":
            return "You are translating from Akan to English. Select the most appropriate English translation from the options provided."
        elif self.type == "context":
            return "You are analyzing an Akan sentence to infer pragmatic context and select the appropriate English translation."
        
    def get_task_prompt(self):

        if self.type == "direct":
            return "Select the best translation by number only. Respond with just the number (1, 2, 3, etc.)."
        elif self.type == "context":
            context_format = """First, infer the pragmatic context by selecting ONE value for each dimension:
            - Audience: [Individual | Small_Group | Large_Group | Broadcast]
            - Status: [Equal | Superior | Subordinate]
            - Age: [Peer | Elder | Younger]
            - Formality: [Formal | Casual]
            - Gender: [Masculine | Feminine | Neutral]
            - Animacy: [Animate | Inanimate]
            - Speech_Act: [Question | Answer | Statement | Command | Request | Greeting]

            Then, based on these inferred tags, select the most appropriate translation by number.

            You MUST respond with valid JSON only. Do not include any text before or after the JSON.
            
            Respond in JSON format:
            {
            "tags": {
                "Audience": "X",
                "Status": "X",
                "Age": "X",
                "Formality": "X",
                "Gender": "X",
                "Animacy": "X",
                "Speech_Act": "X"
            },
            "selection": number
            }
            """

            return context_format
    
    def get_base_prompt(self, akan_sentence: str, english_sentences: list[str]) -> str:
        """
        Construct a prompt that asks the model to pick the best translation.
        
        Parameters
        ----------
        akan_sentence : str
            The Akan sentence to be translated.
        english_sentences : list[str]
            A list of candidate English translations.

        Returns
        -------
        str
            The fully populated prompt string.
        """
        role_block = self.get_role_prompt()
        options_block = self.get_numbered_prompt(english_sentences)
        task_block = self.get_task_prompt()

        # Compose the final prompt
        prompt = f"""{role_block}
        
        Akan sentence: "{akan_sentence}"
        
        Translation options: \n{options_block}
        
        {task_block}
        """
        
        return prompt
    
class FewShotPromptFactory(BasePromptFactory):
    def __init__(self, type):
        self.type = type

    def get_name(self):
        if self.type == "direct":
            return "few_shot-direct"
        elif self.type == "context":
            return "few_shot-context"

    def get_role_prompt(self):
        if self.type == "direct":
            return "You are translating from Akan to English. Select the most appropriate English translation from the options provided."
        elif self.type == "context":
            return "You are analyzing an Akan sentence to infer pragmatic context and select the appropriate English translation."
        
    def get_examples_prompt(self) -> str:
        """Return the example block (differs between direct and context)."""
        if self.type == "direct":
            # Two simple selection examples
            return """Examples:
            Akan: "Ɔyɛ me maame"
            Options: 1. He is my mother 2. She is my mother 3. They are my mother
            Selection: 2

            Akan: "Mema wo akwaaba"
            Options: 1. I welcome you (singular) 2. We welcome you (plural) 3. I welcomed you
            Selection: 1"""
        
        elif self.type == "context":
            # Two richer examples with Analysis + TAGS + SELECTION
            return """Example 1:
            Akan: "Ɔyɛ me mpena"
            Options: 1. He is my boyfriend 2. She is my girlfriend 3. They are my lover
            Analysis: The term "mpena" suggests romantic relationship (casual register), "Ɔ" is 3rd person singular

            First, infer the pragmatic context by selecting ONE value for each dimension:
            - Audience: [Individual | Small_Group | Large_Group | Broadcast]
            - Status: [Equal | Superior | Subordinate]
            - Age: [Peer | Elder | Younger]
            - Formality: [Formal | Casual]
            - Gender: [Masculine | Feminine | Neutral]
            - Animacy: [Animate | Inanimate]
            - Speech_Act: [Question | Answer | Statement | Command | Request | Greeting]

            Then, based on these inferred tags, select the most appropriate translation by number.

            You MUST respond with valid JSON only. Do not include any text before or after the JSON.
            
            Respond in JSON format:
            {
            "tags": {
                "Audience": "Individual",
                "Status": "Equal",
                "Age": "Peer",
                "Formality": "Casual",
                "Gender": "Masculine",
                "Animacy": "Animate",
                "Speech_Act": "Statement"
            },
            "selection": 1
            }

            Example 2:
            Akan: "Yɛfrɛ wo sɛn?"
            Options: 1. What is your name? 2. What do they call you? 3. How do we call you?
            Analysis: Direct question format, likely addressing individual, neutral formality
            
            First, infer the pragmatic context by selecting ONE value for each dimension:
            - Audience: [Individual | Small_Group | Large_Group | Broadcast]
            - Status: [Equal | Superior | Subordinate]
            - Age: [Peer | Elder | Younger]
            - Formality: [Formal | Casual]
            - Gender: [Masculine | Feminine | Neutral]
            - Animacy: [Animate | Inanimate]
            - Speech_Act: [Question | Answer | Statement | Command | Request | Greeting]

            Then, based on these inferred tags, select the most appropriate translation by number.

            You MUST respond with valid JSON only. Do not include any text before or after the JSON.
            
            Respond in JSON format:
            {
            "tags": {
                "Audience": "Individual",
                "Status": "Equal",
                "Age": "Peer",
                "Formality": "Casual",
                "Gender": "Neutral",
                "Animacy": "Animate",
                "Speech_Act": "Question"
            },
            "selection": 1
            }            
            """
        else:
            raise ValueError(f"Unknown prompt type: {self.type}")
    
    def get_task_prompt(self):

        if self.type == "direct":
            return "Select the best translation by number only. Respond with just the number (1, 2, 3, etc.)."
        elif self.type == "context":
            context_format = """First infer the pragmatic context, then select the best translation.
            
            Respond in this format: 
            TAGS: Audience=X, Status=X, Age=X, Formality=X, Gender=X, Animacy=X, Speech_Act=X
            SELECTION: [number]
            """

            return context_format

    def get_base_prompt(self, akan_sentence: str, english_sentences: list[str]) -> str:
        """
        Build a complete few‑shot translation prompt.

        Parameters
        ----------
        akan_sentence : str
            The Akan sentence that the LLM should translate.
        english_sentences : list[str]
            Candidate English translations (any length).

        Returns
        -------
        str
            The full prompt text ready to be sent to the model.
        """
        role_block = self.get_role_prompt()
        options_block = self.get_numbered_prompt(english_sentences)
        examples_block = self.get_examples_prompt()
        task_block = self.get_task_prompt()

        prompt = f"""{role_block}
        {examples_block}

        Now select for this sentence:
        Akan sentence: "{akan_sentence}"
        
        Translation options: \n{options_block}
        
        {task_block}
        """

        return prompt
    
class ChainOfThoughtPrompt:
     def get_base_prompt(akan_sentence: str, english_sentences: list[str]) -> str:
        """
        Build a complete few‑shot translation prompt.

        Parameters
        ----------
        akan_sentence : str
            The Akan sentence that the LLM should translate.
        english_sentences : list[str]
            Candidate English translations (any length).

        Returns
        -------
        str
            The full prompt text ready to be sent to the model.
        """
        # Numbered list of the translation options (tab‑indented for readability)
        numbered_options = [
            f"\t{i}. {translation}"
            for i, translation in enumerate(english_sentences, start=1)
        ]
        options_block = "\n".join(numbered_options)

        prompt = f"""
        You are translating from Akan to English. Follow these reasoning steps to select the most appropriate translation:

        Akan sentence: "{akan_sentence}"

        Translation options: \n{options_block}

        Step 1: Analyze the Akan sentence structure and identify key linguistic features.
        Step 2: Consider what each translation option implies about the context.
        Step 3: Determine which option best matches the likely intended meaning.
        Step 4: Select the best translation by number.

        Provide your reasoning for steps 1-3, then state your final selection as "SELECTION: [number]"
        """

        return prompt