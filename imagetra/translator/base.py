from imagetra.common.base import Base
from imagetra.common.logger import get_logger
from imagetra.common.batch import get_batch

from typing import List

logger = get_logger('Translator')

class BaseTranslator(Base):
    def __init__(self, subbatch_size: int=10, cache: bool=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.subbatch_size = subbatch_size
        self.translation_cache = {}
        self.cache = cache

    def contextualize(self, texts: List[str]) -> str:
        return ' '.join(texts)

    def decontextualize(self, text: str) -> List[str]:
        return [text]

    def forward(self, texts: List[str], src_lang: str=None, trg_lang: str=None, **kargs) -> List[str]:
        def format_key(key):
            return f"{src_lang}_{trg_lang}_{key}"

        if not self.cache:
            output = []
            for batch in get_batch(texts, self.subbatch_size, show_pbar=self.show_pbar):
                output += self.translate(batch, src_lang=src_lang, trg_lang=trg_lang, **kargs)
            return output

        texts_set = [text for text in set(texts) if text not in self.translation_cache]
        translate_map = {}
        for batch in get_batch(texts_set, self.subbatch_size, show_pbar=self.show_pbar):
            batch_outputs = self.translate(batch, src_lang=src_lang, trg_lang=trg_lang, **kargs)
            translate_map.update({format_key(text): output for text, output in zip(batch, batch_outputs)})
        self.translation_cache.update(translate_map)

        output = []
        for text in texts:
            key = format_key(text)
            translation = translate_map[key] if key in translate_map else self.translation_cache[key]
            output.append(translation)
        return output
        
    def translate(self, texts: List[str], src_lang: str=None, trg_lang: str=None, **kargs) -> List[str]:
        raise NotImplementedError()

