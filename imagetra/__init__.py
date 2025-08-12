import pkgutil, importlib

__all__ = []
package_name = __name__

# Import all submodules in the current package
for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    # Skip private modules
    if module_name.startswith('_'):
        continue

    full_module_name = f"{package_name}.{module_name}"
    module = importlib.import_module(full_module_name)
    globals()[module_name] = module
    __all__.append(module_name)

def build_recodetector(configs):
    if configs.recodetector_name == 'doctr':
        from imagetra.detector.doctr import DoctrRecoDetector
        kwargs = {}
        if configs.recodetector_detector_name is not None:
            kwargs['detector_name'] = configs.recodetector_detector_name
        if configs.recodetector_recognizer_name is not None:
            kwargs['recognizer_name'] = configs.recodetector_recognizer_name
        recodetector = DoctrRecoDetector(**kwargs)
    elif configs.recodetector_name == 'openocr':
        from imagetra.detector.openocr import OpenOCRRecoDetector
        recodetector = OpenOCRRecoDetector()
    elif configs.recodetector_name == 'easyocr':
        from imagetra.detector.easyocr import EasyOCRRecoDetector
        import re
        langs = re.split(f' *, *', configs.recodetector_langs)
        recodetector = EasyOCRRecoDetector(langs=langs)
    elif configs.recodetector_name == 'paddleocr':
        from imagetra.detector.paddleocr import PaddleOCRRecoDetector
        recodetector = PaddleOCRRecoDetector(lang=configs.recodetector_langs)
    else:
        raise NotImplementedError(f'Unknown {configs.recodetector_name}')
    if configs.common_gpu:
        recodetector.to('cuda')
    return recodetector

def build_translator(configs):
    if configs.translator_name == 'nllb':
        from imagetra.translator.nllb import NLLBTranslator
        translator = NLLBTranslator(
            configs.translator_model_name,
            trg_lang=configs.translator_trg_lang
        )
    elif configs.translator_name == 'textra':
        from imagetra.translator.textra import TexTraTranslator
        translator = TexTraTranslator(
            user_id=configs.translator_textra_user_id,
            api_key=configs.translator_textra_api_key,
            api_secret=configs.translator_textra_api_secret,
            src_lang='en',
            trg_lang=configs.translator_trg_lang
        )
    else:
        raise NotImplementedError(f'Unknown {configs.translator_name}')
    if configs.common_gpu:
        translator.to('cuda')
    return translator

def build_editor(configs):
    if configs.editor_name == 'srnet':
        from imagetra.editor.srnet import SRNetEditor
        editor = SRNetEditor(
            model_path=configs.editor_model_path,
            font=configs.editor_font_path,
            min_font_size=configs.editor_min_font_size,
            padding_lr=configs.editor_bbox_margin,
            padding_tb=configs.editor_bbox_margin,
        )
    elif configs.editor_name == 'render':
        from imagetra.editor.render import RenderEditor
        editor = RenderEditor(
            font=configs.editor_font_path,
            min_font_size=configs.editor_min_font_size,
            padding_lr=configs.editor_bbox_margin,
            padding_tb=configs.editor_bbox_margin,
            num_workers=configs.editor_num_workers,
        )
    else:
        raise NotImplementedError(f'Unknown {configs.editor_name}')
    if configs.common_gpu:
        editor.to('cuda')
    return editor

def build_pipeline(name, configs, logfile=None):
    recodetector = build_recodetector(configs)
    translator = build_translator(configs)
    editor = build_editor(configs) if name != 'img2txt' else None

    if name == 'vdo2vdo':
        from imagetra.pipeline.vdo2vdo import Video2Video
        pipeline = Video2Video(
            recodetector=recodetector,
            editor=editor,
            translator=translator,
            margin=configs.editor_bbox_margin,
            logfile=logfile,
            min_share_ratio=0.5,
            fix_bbox=True if configs.editor_name == 'srnet' else False
        )
    elif name == 'img2img':
        from imagetra.pipeline.img2img import Image2Image
        pipeline = Image2Image(
            recodetector=recodetector,
            editor=editor,
            translator=translator,
            margin=configs.editor_bbox_margin,
            logfile=logfile,
            fix_bbox=True if configs.editor_name == 'srnet' else False,
        )
    elif name == 'img2txt':
        from imagetra.pipeline.img2txt import Image2Text
        pipeline = Image2Text(
            recodetector=recodetector,
            translator=translator,
            logfile=logfile
        )
    else:
        raise NotImplementedError(f'Unknown {name}')

    return pipeline