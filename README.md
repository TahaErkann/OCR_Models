# OCR Model Karşılaştırma Sistemi

Modüler ve esnek bir OCR (Optical Character Recognition) model karşılaştırma sistemi. Farklı platformlardaki (HuggingFace, Ollama, PaddleOCR) OCR modellerini kolayca karşılaştırabilirsiniz.

## 🎯 Özellikler

- **Modüler Yapı**: Yeni modeller kolayca eklenebilir
- **Çoklu Platform Desteği**: HuggingFace, Ollama, PaddleOCR
- **Görsel Karşılaştırma**: Streamlit arayüzü ile interaktif test
- **Performans Metrikleri**: Güven skoru, işlem süresi, kelime sayısı
- **Bounding Box Görselleştirme**: Tespit edilen metinlerin görsel gösterimi

## 📋 Gereksinimler

- Python 3.8+
- CUDA (opsiyonel, GPU için)

## 🚀 Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd ocr_models
```

2. **Sanal ortam oluşturun (önerilir):**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **HuggingFace Token Kurulumu (Gated Modeller İçin):**
   - Qwen3-VL-2B-Instruct gibi gated modeller için token gereklidir
   - Detaylı kurulum için: [HUGGINGFACE_TOKEN_SETUP.md](HUGGINGFACE_TOKEN_SETUP.md)
   - Hızlı kurulum:
   ```bash
   # Windows (CMD)
   set HF_TOKEN=hf_xxxxxxxxxxxxx
   
   # Windows (PowerShell)
   $env:HF_TOKEN="hf_xxxxxxxxxxxxx"
   
   # Linux/Mac
   export HF_TOKEN="hf_xxxxxxxxxxxxx"
   ```
   - Token almak için: https://huggingface.co/settings/tokens

5. **PaddleOCR (opsiyonel):**
```bash
pip install paddleocr paddlepaddle
```

6. **Ollama kurulumu (opsiyonel):**
- [Ollama](https://ollama.ai) indirin ve kurun
- İstediğiniz modeli yükleyin:
```bash
ollama pull qwen3-vl:2b
```

## 🎮 Kullanım

### Streamlit Uygulamasını Başlatma

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak açılacaktır (genellikle http://localhost:8501)

### Yeni Model Ekleme

1. `config.py` dosyasını açın
2. `MODEL_CONFIGS` listesine yeni model yapılandırması ekleyin:

```python
{
    "id": "yeni-model-id",
    "platform": "huggingface",  # veya "ollama", "paddleocr"
    "model_name": "model/adı",
    "enabled": True,
    "description": "Model açıklaması"
}
```

3. Eğer yeni bir platform ekliyorsanız:
   - `models/` klasörüne yeni bir implementasyon dosyası ekleyin
   - `BaseOCRModel` sınıfından türetin
   - `models/registry.py` dosyasına platform'u kaydedin

## 📁 Proje Yapısı

```
ocr_models/
├── models/              # OCR model implementasyonları
│   ├── base.py         # Temel OCR sınıfı
│   ├── huggingface_ocr.py
│   ├── ollama_ocr.py
│   └── registry.py     # Model kayıt sistemi
├── utils/              # Yardımcı fonksiyonlar
│   ├── image_processor.py
│   └── comparison.py
├── app.py              # Streamlit ana uygulama
├── config.py           # Yapılandırma dosyası
├── requirements.txt    # Python bağımlılıkları
└── README.md
```

## 🔧 Yapılandırma

`config.py` dosyasında şu ayarları yapabilirsiniz:

- **MODEL_CONFIGS**: Model yapılandırmaları
- **STREAMLIT_CONFIG**: Streamlit ayarları
- **IMAGE_PROCESSING_CONFIG**: Görüntü işleme ayarları
- **MODEL_INIT_CONFIG**: Model başlatma ayarları

## 📊 Kullanım Senaryoları

### 1. Tek Model Testi
- Bir model seçin
- Görüntü yükleyin
- OCR sonuçlarını görüntüleyin

### 2. Model Karşılaştırması
- Birden fazla model seçin
- Aynı görüntüyü tüm modellerde test edin
- Performans metriklerini karşılaştırın

### 3. Yeni Model Entegrasyonu
- Yeni bir OCR modeli geldiğinde
- Sadece yeni bir sınıf yazın ve kaydedin
- Otomatik olarak sistemde kullanılabilir hale gelir

## 🛠️ Geliştirme

### Yeni Platform Ekleme

1. `models/` klasörüne yeni dosya ekleyin (örn: `new_platform_ocr.py`)
2. `BaseOCRModel` sınıfından türetin
3. Gerekli metodları implement edin:
   - `initialize()`
   - `predict(image)`
   - `get_model_info()`
4. `models/registry.py` dosyasına platform'u kaydedin

### Örnek Yeni Model Implementasyonu

```python
from models.base import BaseOCRModel, OCRResult
from PIL import Image

class MyCustomOCR(BaseOCRModel):
    def __init__(self, model_name: str):
        super().__init__(model_name, "myplatform")
    
    def initialize(self) -> bool:
        # Model yükleme kodu
        self.initialized = True
        return True
    
    def predict(self, image: Image.Image) -> OCRResult:
        # OCR işlemi
        return OCRResult(text="...", confidence=0.95)
    
    def get_model_info(self) -> Dict:
        return {"model_name": self.model_name, ...}
```

## 📝 Notlar

- HuggingFace modelleri için GPU önerilir
- Ollama için yerel sunucunun çalışıyor olması gerekir
- PaddleOCR opsiyonel bir bağımlılıktır
- Büyük modeller için yeterli RAM/VRAM gerekir

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje açık kaynaklıdır.

## 🙏 Teşekkürler

- HuggingFace ekibine
- Ollama ekibine
- PaddleOCR ekibine

