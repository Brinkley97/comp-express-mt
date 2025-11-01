
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