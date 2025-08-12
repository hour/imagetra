from imagetra.translator.base import BaseTranslator
from imagetra.common.lang import to_nll_lang

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List

class NLLBTranslator(BaseTranslator):
    MODELS = [
        "facebook/nllb-200-distilled-600M",
        "facebook/nllb-200-distilled-1.3B",
        "facebook/nllb-200-1.3B",
        "facebook/nllb-200-3.3B",
        "facebook/nllb-moe-54b",
    ]
    
    def __init__(self, name, trg_lang: str=None, **kwargs) -> None:
        super().__init__(**kwargs)
        assert(name in self.MODELS)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
        self.device = 'cpu'
        self.trg_lang = to_nll_lang(trg_lang) if trg_lang is not None else None

    def to(self, device: str):
        self.device = device
        self.model.to(self.device)

    def translate(self, texts: List[str], src_lang: str=None, trg_lang: str=None):
        if trg_lang is None:
            trg_lang = self.trg_lang
        else:
            trg_lang = to_nll_lang(trg_lang)
        assert(trg_lang is not None), 'Specify `trg_lang`.'
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(trg_lang), max_length=30
        )
        outputs = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return outputs
        
