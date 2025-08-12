import langcodes

def to_nll_lang(lang):
    lang = langcodes.get(lang)
    lang = lang.maximize()
    lang = langcodes.get(lang)
    return f'{lang.to_alpha3()}_{lang.script}'
