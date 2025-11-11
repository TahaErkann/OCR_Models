"""
Image Processing Utilities
Görüntü ön işleme ve görselleştirme fonksiyonları
"""

from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class ImageProcessor:
    """Görüntü işleme yardımcı sınıfı"""
    
    @staticmethod
    def preprocess_image(
        image: Image.Image,
        resize: Optional[Tuple[int, int]] = None,
        enhance_contrast: bool = False,
        convert_grayscale: bool = False
    ) -> Image.Image:
        """
        Görüntüyü ön işle
        
        Args:
            image: PIL Image
            resize: Yeniden boyutlandırma (width, height)
            enhance_contrast: Kontrast artırma
            convert_grayscale: Gri tonlamaya çevir
            
        Returns:
            Image.Image: Ön işlenmiş görüntü
        """
        processed = image.copy()
        
        # Gri tonlamaya çevir
        if convert_grayscale:
            processed = processed.convert('L')
            processed = processed.convert('RGB')  # OCR modelleri genelde RGB bekler
        
        # Kontrast artırma
        if enhance_contrast:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(1.5)
        
        # Yeniden boyutlandırma
        if resize:
            processed = processed.resize(resize, Image.Resampling.LANCZOS)
        
        return processed
    
    @staticmethod
    def draw_bounding_boxes(
        image: Image.Image,
        bboxes: List[Tuple[int, int, int, int]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        line_width: int = 2
    ) -> Image.Image:
        """
        Görüntü üzerine bounding box çiz
        
        Args:
            image: PIL Image
            bboxes: Bounding box listesi [(x1, y1, x2, y2), ...]
            labels: Her bbox için etiket listesi
            colors: Her bbox için renk listesi (hex veya isim)
            line_width: Çizgi kalınlığı
            
        Returns:
            Image.Image: Bounding box'ları çizilmiş görüntü
        """
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # Varsayılan renkler
        default_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            
            # Renk seç
            if colors and idx < len(colors):
                color = colors[idx]
            else:
                color = default_colors[idx % len(default_colors)]
            
            # Dikdörtgen çiz
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            # Etiket ekle
            if labels and idx < len(labels):
                label = labels[idx]
                # Etiket arka planı için küçük bir kutu çiz
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                
                # Etiket metni boyutunu hesapla
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # Etiket arka planı
                label_y = max(0, y1 - text_height - 4)
                draw.rectangle(
                    [x1, label_y, x1 + text_width + 4, label_y + text_height + 4],
                    fill=color,
                    outline=color
                )
                draw.text((x1 + 2, label_y + 2), label, fill='white', font=font)
        
        return result_image
    
    @staticmethod
    def draw_multiple_model_results(
        image: Image.Image,
        results: Dict[str, Dict],
        model_colors: Optional[Dict[str, str]] = None
    ) -> Image.Image:
        """
        Birden fazla modelin sonuçlarını aynı görüntüde göster
        
        Args:
            image: PIL Image
            results: Model sonuçları dict'i
                {
                    "model_id": {
                        "bboxes": [...],
                        "words": [...]
                    }
                }
            model_colors: Her model için renk dict'i
            
        Returns:
            Image.Image: Tüm sonuçları gösteren görüntü
        """
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # Varsayılan renkler
        default_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        if model_colors is None:
            model_colors = {}
        
        model_list = list(results.keys())
        
        for model_idx, model_id in enumerate(model_list):
            model_result = results[model_id]
            
            # Model rengi
            color = model_colors.get(model_id, default_colors[model_idx % len(default_colors)])
            
            # Bbox'ları çiz
            bboxes = model_result.get("bboxes", [])
            words = model_result.get("words", [])
            
            # Eğer words varsa, onları kullan
            if words:
                for word in words:
                    if "bbox" in word:
                        bbox = word["bbox"]
                        x1, y1, x2, y2 = bbox
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            else:
                # Sadece bbox'lar varsa
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        return result_image
    
    @staticmethod
    def create_comparison_image(
        original_image: Image.Image,
        model_results: Dict[str, Dict],
        model_colors: Optional[Dict[str, str]] = None
    ) -> Image.Image:
        """
        Karşılaştırma görüntüsü oluştur (her model için ayrı görüntü)
        
        Args:
            original_image: Orijinal görüntü
            model_results: Model sonuçları
            model_colors: Model renkleri
            
        Returns:
            Image.Image: Karşılaştırma görüntüsü (yan yana)
        """
        from PIL import Image as PILImage
        
        num_models = len(model_results)
        if num_models == 0:
            return original_image
        
        # Her model için görüntü oluştur
        images = []
        model_ids = list(model_results.keys())
        
        default_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        if model_colors is None:
            model_colors = {}
        
        for model_id in model_ids:
            model_result = model_results[model_id]
            color = model_colors.get(model_id, default_colors[len(images) % len(default_colors)])
            
            # Bu model için görüntü oluştur
            img = original_image.copy()
            draw = ImageDraw.Draw(img)
            
            bboxes = model_result.get("bboxes", [])
            words = model_result.get("words", [])
            
            if words:
                for word in words:
                    if "bbox" in word:
                        bbox = word["bbox"]
                        x1, y1, x2, y2 = bbox
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            else:
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            images.append(img)
        
        # Görüntüleri yan yana birleştir
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        
        comparison_image = PILImage.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            comparison_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        return comparison_image

