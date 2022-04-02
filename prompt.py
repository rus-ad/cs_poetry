#!/usr/bin/env python
# coding: utf-8
# %%
from ruprompts import PromptFormat
from ruprompts import TensorPromptProvider

from ruprompts import Prompt
from ruprompts import Text2TextPreprocessor


# %%
class BasePrompt:
    
    def __init__(self, prompt_format: str, model: dict):
        self.prompt_format = PromptFormat(prompt_format)
        self.model = model['model']
        self.tokenizer = model['tokenizer']
        self.preprocessor = None
        
    def set_preprocessor(self, target_field, truncation_field):
        self.prompt_provider = TensorPromptProvider()
        self.prompt = Prompt(self.prompt_format, self.prompt_provider)
        self.prompt.patch(self.model, self.tokenizer)
        self.preprocessor = Text2TextPreprocessor(
            prompt_format=self.prompt_format,
            tokenizer=self.tokenizer,
            target_field=target_field,
            max_tokens=1792,
            truncation_field=truncation_field,
        )
        
    def preprocess_dataset(self, data):
        if self.preprocessor is None:
            raise NotImplementedError('preprocessor is None')
        return data.map(self.preprocessor)

