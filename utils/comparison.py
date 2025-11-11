"""
Model Comparison Utilities
OCR modellerini karşılaştırma fonksiyonları
"""

from typing import Dict, List, Optional
from PIL import Image
import time
from models.base import BaseOCRModel, OCRResult


class ModelComparator:
    """OCR modellerini karşılaştırma sınıfı"""
    
    @staticmethod
    def compare_models(
        models: Dict[str, BaseOCRModel],
        image: Image.Image,
        include_timing: bool = True
    ) -> Dict[str, Dict]:
        """
        Birden fazla modeli aynı görüntüde karşılaştır
        
        Args:
            models: Model dict'i {model_id: model_instance}
            image: Test görüntüsü
            include_timing: Zamanlama bilgisi dahil et
            
        Returns:
            Dict: Karşılaştırma sonuçları
                {
                    "model_id": {
                        "result": OCRResult,
                        "time": float,
                        "success": bool,
                        "error": str
                    }
                }
        """
        results = {}
        
        for model_id, model in models.items():
            if not model.is_initialized():
                results[model_id] = {
                    "result": None,
                    "time": 0.0,
                    "success": False,
                    "error": "Model başlatılmamış"
                }
                continue
            
            try:
                start_time = time.time()
                ocr_result = model.predict(image)
                elapsed_time = time.time() - start_time
                
                result_data = {
                    "result": ocr_result,
                    "time": elapsed_time if include_timing else None,
                    "success": True,
                    "error": None
                }
                
                results[model_id] = result_data
                
            except Exception as e:
                results[model_id] = {
                    "result": None,
                    "time": 0.0,
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    @staticmethod
    def calculate_metrics(results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Model metriklerini hesapla
        
        Args:
            results: compare_models() sonuçları
            
        Returns:
            Dict: Metrikler
                {
                    "model_id": {
                        "text_length": int,
                        "word_count": int,
                        "avg_confidence": float,
                        "processing_time": float,
                        "success_rate": float
                    }
                }
        """
        metrics = {}
        
        for model_id, result_data in results.items():
            if not result_data.get("success"):
                metrics[model_id] = {
                    "text_length": 0,
                    "word_count": 0,
                    "avg_confidence": 0.0,
                    "processing_time": 0.0,
                    "success_rate": 0.0
                }
                continue
            
            ocr_result = result_data.get("result")
            if ocr_result is None:
                continue
            
            text = ocr_result.text
            words = ocr_result.words
            
            # Metrikler
            text_length = len(text)
            word_count = len(words) if words else len(text.split())
            avg_confidence = ocr_result.confidence
            processing_time = result_data.get("time", 0.0)
            
            # Kelime bazlı güven skoru varsa ortalama al
            if words:
                confidences = [w.get("confidence", 0.0) for w in words]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
            
            metrics[model_id] = {
                "text_length": text_length,
                "word_count": word_count,
                "avg_confidence": avg_confidence,
                "processing_time": processing_time,
                "success_rate": 1.0 if result_data.get("success") else 0.0
            }
        
        return metrics
    
    @staticmethod
    def get_best_model(
        results: Dict[str, Dict],
        metric: str = "avg_confidence"
    ) -> Optional[str]:
        """
        En iyi modeli seç
        
        Args:
            results: calculate_metrics() sonuçları
            metric: Karşılaştırma metrikleri ('avg_confidence', 'word_count', 'text_length')
            
        Returns:
            str: En iyi model ID'si
        """
        if not results:
            return None
        
        valid_results = {
            k: v for k, v in results.items()
            if v.get("success_rate", 0) > 0
        }
        
        if not valid_results:
            return None
        
        # Metrik değerine göre sırala
        sorted_models = sorted(
            valid_results.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )
        
        return sorted_models[0][0] if sorted_models else None

