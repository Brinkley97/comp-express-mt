
class ZeroShotPrompt():
    def get_base_prompt(akan_sentence: str, english_sentences: list[str]) -> str:
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
        # Build a numbered list from the translations
        numbered_options = [
            f"\t{i}. {translation}"          # i starts at 1
            for i, translation in enumerate(english_sentences, start=1)
        ]

        # Join the list into a single block of text, each option on its own line
        options_block = "\n".join(numbered_options)

        # Compose the final prompt
        prompt = f"""
        You are translating from Akan to English. Select the most appropriate English translation from the options provided.
        
        Akan sentence: "{akan_sentence}"
        
        Translation options: \n{options_block}
        
        Select the best translation by number only. Respond with just the number (1, 2, 3, …).
        """
        
        return prompt
    

class FewShotPrompt:
    @staticmethod
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
        
        Translation options: \n{options_block}
        
        Select the best translation by number only. Respond with just the number (1, 2, 3, etc.). You must choose a number and nothing else."""

        return prompt