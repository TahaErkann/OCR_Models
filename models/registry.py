"""
Model Registry
OCR modellerini kaydetme ve yönetme sistemi
"""

from typing import Dict, List, Optional, Type
from .base import BaseOCRModel
from .huggingface_ocr import HuggingFaceOCR, PaddleOCRWrapper
from .ollama_ocr import OllamaOCR


class ModelRegistry:
    """OCR modellerini kaydetme ve yönetme sınıfı"""
    
    _instance = None
    _models: Dict[str, BaseOCRModel] = {}
    _model_classes: Dict[str, Type[BaseOCRModel]] = {
        "huggingface": HuggingFaceOCR,
        "ollama": OllamaOCR,
        "paddleocr": PaddleOCRWrapper
    }
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def register_model(
        self,
        model_id: str,
        platform: str,
        model_name: str,
        **kwargs
    ) -> bool:
        """
        Yeni bir model kaydet
        
        Args:
            model_id: Model için benzersiz ID
            platform: Platform tipi ('huggingface', 'ollama', 'paddleocr')
            model_name: Model adı
            **kwargs: Model'e özel parametreler (token, device, base_url, vb.)
            
        Returns:
            bool: Başarılı ise True
        """
        try:
            if platform not in self._model_classes:
                print(f"Bilinmeyen platform: {platform}")
                return False
            
            # Model sınıfını al
            model_class = self._model_classes[platform]
            
            # Model instance oluştur
            # kwargs içinde token, device, base_url gibi parametreler olabilir
            model = model_class(model_name=model_name, **kwargs)
            
            # Kaydet
            self._models[model_id] = model
            
            print(f"Model kaydedildi: {model_id} ({platform})")
            return True
            
        except Exception as e:
            print(f"Model kayıt hatası: {str(e)}")
            return False
    
    def get_model(self, model_id: str) -> Optional[BaseOCRModel]:
        """Model ID ile model al"""
        return self._models.get(model_id)
    
    def get_all_models(self) -> Dict[str, BaseOCRModel]:
        """Tüm modelleri döndür"""
        return self._models.copy()
    
    def get_model_ids(self) -> List[str]:
        """Tüm model ID'lerini döndür"""
        return list(self._models.keys())
    
    def initialize_model(self, model_id: str) -> bool:
        """Belirli bir modeli başlat"""
        model = self.get_model(model_id)
        if model is None:
            print(f"Model bulunamadı: {model_id}")
            return False
        
        if model.is_initialized():
            print(f"Model zaten başlatılmış: {model_id}")
            return True
        
        return model.initialize()
    
    def initialize_all_models(self) -> Dict[str, bool]:
        """Tüm modelleri başlat"""
        results = {}
        for model_id in self._models.keys():
            results[model_id] = self.initialize_model(model_id)
        return results
    
    def remove_model(self, model_id: str) -> bool:
        """Modeli kayıttan kaldır"""
        if model_id in self._models:
            del self._models[model_id]
            print(f"Model kaldırıldı: {model_id}")
            return True
        return False
    
    def register_custom_model_class(
        self,
        platform: str,
        model_class: Type[BaseOCRModel]
    ):
        """
        Özel model sınıfı kaydet (yeni platformlar için)
        
        Args:
            platform: Platform adı
            model_class: BaseOCRModel'den türeyen sınıf
        """
        if not issubclass(model_class, BaseOCRModel):
            raise ValueError("Model sınıfı BaseOCRModel'den türemeli")
        
        self._model_classes[platform] = model_class
        print(f"Özel model sınıfı kaydedildi: {platform}")

