"""
Base OCR Model Interface
Tüm OCR modellerinin uyacağı temel sınıf
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np


class OCRResult:
    """OCR sonuçlarını tutan veri yapısı"""
    
    def __init__(
        self,
        text: str,
        confidence: float = 1.0,
        bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
        words: Optional[List[Dict]] = None
    ):
        """
        Args:
            text: Tespit edilen metin
            confidence: Güven skoru (0-1 arası)
            bboxes: Bounding box koordinatları [(x1, y1, x2, y2), ...]
            words: Kelime bazlı detaylar [{"text": "...", "bbox": [...], "confidence": ...}, ...]
        """
        self.text = text
        self.confidence = confidence
        self.bboxes = bboxes or []
        self.words = words or []
    
    def to_dict(self) -> Dict:
        """Sonuçları dictionary formatına dönüştür"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bboxes": self.bboxes,
            "words": self.words
        }


class BaseOCRModel(ABC):
    """Tüm OCR modellerinin türeyeceği temel sınıf"""
    
    def __init__(self, model_name: str, platform: str):
        """
        Args:
            model_name: Model adı
            platform: Platform adı (huggingface, ollama, vb.)
        """
        self.model_name = model_name
        self.platform = platform
        self.initialized = False
        self.model = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Modeli yükle ve başlat
        
        Returns:
            bool: Başarılı ise True
        """
        pass
    
    @abstractmethod
    def predict(self, image: Image.Image) -> OCRResult:
        """
        Görüntü üzerinde OCR işlemi yap
        
        Args:
            image: PIL Image nesnesi
            
        Returns:
            OCRResult: OCR sonuçları
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """
        Model hakkında bilgi döndür
        
        Returns:
            Dict: Model bilgileri
        """
        pass
    
    def is_initialized(self) -> bool:
        """Model başlatılmış mı kontrol et"""
        return self.initialized
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Görüntüyü ön işleme tabi tut (override edilebilir)
        
        Args:
            image: PIL Image nesnesi
            
        Returns:
            Image.Image: Ön işlenmiş görüntü
        """
        # Varsayılan olarak görüntüyü olduğu gibi döndür
        return image
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', platform='{self.platform}')"

