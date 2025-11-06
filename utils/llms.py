"""
Code belongs to: https://github.com/ufdatastudio/predictions

Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

Factory Method Design Pattern (https://refactoring.guru/design-patterns/factory-method/python/example#lang-features)
"""

import os
import openai
import pathlib
import torch
import ipdb

import pandas as pd

from groq import Groq
from tqdm import tqdm
from typing import Dict, List

from dotenv import load_dotenv
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()  # Load environment variables from .env file

class TextGenerationModelFactory(ABC):
    """An abstract base class to load any pre-trained generation model"""
    
    def __init__(self):
        """In the init method (also called constructor), initialize our class with variables or attributes."""
        # Create instance variables or attributes
        # Standardized model parameters
        self.temperature = 0.6
        self.top_p = 0.9
        self.model_name = None
   
    def map_platform_to_api(self, platform_name: str):
        """
        Parameter:
        ----------
        platform_name : `str`
            Platform to use for generations.
        
        Returns:
        --------
        api_key : `str`
            The api key of specified platform.

        """
        platform_to_api_mappings = {
            "GROQ_CLOUD" : os.getenv('GROQ_CLOUD_API_KEY'), # https://console.groq.com/docs/models
            "NAVI_GATOR" : os.getenv('NAVI_GATOR_API_KEY'), # https://it.ufl.edu/ai/navigator-toolkit/
            "HUGGING_FACE": os.getenv('HUGGING_FACE_API_KEY') # https://huggingface.co/models?pipeline_tag=text-generation&sort=trending
        }

        api_key = platform_to_api_mappings.get(platform_name)
        
        if api_key is None:
            raise ValueError("API_KEY environment variable not set")
        
        return api_key
    
    @classmethod        
    def create_instance(self, model_name):

        if model_name == 'llama-3.1-70b-instruct':
            return Llama3170BInstructTextGenerationModel()
        elif model_name == 'llama-3.3-70b-instruct':
            return Llama3370BInstructTextGenerationModel()
        elif model_name == 'mixtral-8x7b-instruct':
            return Mixtral87BInstructTextGenerationModel()
        elif model_name == 'llama-3.1-8b-instruct':
            return Llama318BInstructTextGenerationModel()
        elif model_name == 'mistral-7b-instruct':
            return Mistral7BInstructTextGenerationModel()     
        elif model_name == 'mistral-small-3.1':
            return MistralSmall31TextGenerationModel()
        else:
            raise ValueError(f"Unknown class name: {model_name}")

    def assistant(self, content: str) -> Dict:
        """Create an assistant message.
        
        Parameters:
        -----------
        content: `str`
            The content of the assistant message.
        
        Returns:
        --------
        Dict
            A dictionary representing the assistant message.
        """

        return {"role": "assistant", "content": content}
    
    def user(self, content: str) -> Dict:
        """Create a user message.
        
        Parameters:
        -----------
        content : `str`
            The content of the user message.
        
        Returns:
        --------
        Dict
            A dictionary representing the user message.
        """

        return {"role": "user", "content": content}
    
    def chat_completion(self, messages: List[Dict]) -> str:
        """Generate a chat completion response.
        
        Parameters:
        -----------
        messages: `List[Dict]`
            A list of dictionaries representing the chat history.
        
        model: `str`
            The name of the model to use.
        
        temperature: `float`
            Sampling temperature.
        
        top_p: `float`
            Nucleus sampling parameter.
        
        Returns:
        --------
        `str`
            The generated chat completion response.
        """

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content
    
    def generate(self, prompt: str) -> pd.DataFrame:
        """Generate a completion response and return as a DataFrame.

        Parameters
        ----------
        prompt: `str`
            The prompt to generate data.

        Returns
        -------
        `pd.DataFrame`
            The generated completion response formatted as a DataFrame.
        """
        # Generate the raw prediction text
        # print(f"prompt:\n\t{prompt}")
        raw_text = self.chat_completion([self.user(prompt)])
        # print(f"\t{self.model_name} generated: {raw_text}\n")
        # print(f"generates:\n{raw_text}")
        
        # Parse the raw text into structured data (assuming a consistent format)
        for line in raw_text.split("\n"):
            if line.strip():  # Skip empty lines
                return line.strip()

    def __name__(self):
        pass

class Llama3170BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.1-70b-instruct"

class Llama3370BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.3-70b-instruct"

class Mixtral87BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()
    
    def __name__(self):
        return "mixtral-8x7b-instruct"    

class Llama318BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.1-8b-instruct"

class Mistral7BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "mistral-7b-instruct"
    
class MistralSmall31TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "mistral-small-3.1"