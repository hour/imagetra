from dataclasses import dataclass
import yaml, os

@dataclass
class Config:
    recodetector_name: str="paddleocr"
    recodetector_detector_name: str=None
    recodetector_recognizer_name: str=None
    recodetector_langs: str='en'
    recodetector_detect_min_score: float=0
    recodetector_recognize_min_score: float=0
    recodetector_min_width: float=26
    recodetector_min_height: float=26
    editor_name: str="render"
    editor_model_path: str=None
    editor_font_path: str=None
    editor_min_font_size: int=16
    editor_bbox_margin: float=0
    translator_name: str='nllb'
    translator_model_name: str='facebook/nllb-200-distilled-600M'
    translator_trg_lang: str='jpn_Jpan'
    translator_textra_user_id: str=None
    translator_textra_api_key: str=None
    translator_textra_api_secret: str=None
    common_gpu: bool=True
    common_greedy: bool=False
    common_num_workers: int=1
    
    @classmethod
    def load_yaml(cls, path: str):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        
        homedir = os.path.dirname(path)
        flat_config = {}
        for module_name, properties in config.items():
            for key, value in properties.items():
                if key.endswith("_path") and not os.path.isabs(value):
                    value = os.path.join(homedir, value)
                flat_config[f'{module_name}_{key}'] = value
        return cls(**flat_config)
    