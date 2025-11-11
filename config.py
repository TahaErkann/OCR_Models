"""
Configuration File
OCR modellerinin yapılandırma dosyası
"""

# HuggingFace Token (gated modeller için gerekli)
# Token'ı buraya ekleyebilir veya environment variable olarak kullanabilirsiniz
# Environment variable: HF_TOKEN veya HUGGINGFACE_TOKEN
HUGGINGFACE_TOKEN = "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"  # Varsayılan None, environment variable'dan alınacak
# Örnek: HUGGINGFACE_TOKEN = "hf_xxxxxxxxxxxxx"

# OCR Prompt - Tüm modeller için ortak prompt
# Türkçe metinler için optimize edilmiş, detaylı ama net talimatlar içerir
OCR_PROMPT = (
    "Bu görüntüdeki tüm metinleri çıkar. "
    "Her kelimeyi, sayıyı ve karakteri tam olarak göründüğü gibi oku. "
    "Satır sonlarını ve boşlukları koru. "
    "Türkçe karakterleri doğru şekilde kullan (ı, ş, ğ, ü, ö, ç, İ, Ş, Ğ, Ü, Ö, Ç). "
    "Büyük/küçük harf ayrımına dikkat et. "
    "Noktalama işaretlerini ve özel karakterleri koru. "
    "Sadece çıkarılan metni döndür, başka hiçbir açıklama veya yorum ekleme."
)

# Model yapılandırmaları
MODEL_CONFIGS = [
    #{
        #"id": "qwen3-vl:2b",
        #"platform": "huggingface",
        #"model_name": "Qwen/Qwen3-VL-2B-Instruct",
        #"token": None,  # None ise environment variable'dan alınacak, veya token'ı buraya yazın
        #"enabled": True,
        #"description": "Qwen3-VL 2B modeli (HuggingFace - Lokal)"
    #},
    {
        "id": "qwen3-vl:4b-ollama",
        "platform": "ollama",
        "model_name": "qwen3-vl:4b",
        "base_url": "http://localhost:11434",
        "enabled": True,
        "description": "Qwen3-VL 4B modeli (Ollama)"
    },
    {
        "id": "paddleocr",
        "platform": "paddleocr",
        "model_name": "paddleocr",
        "enabled": True,
        "description": "PaddleOCR modeli"
    },
    # Yeni modeller buraya eklenebilir
    # {
    #     "id": "new-model",
    #     "platform": "huggingface",
    #     "model_name": "model/path",
    #     "enabled": True,
    #     "description": "Yeni model açıklaması"
    # }
    {
        "id":"Qwen2.5-VL-3B-Instruct",
        "platform": "ollama",
        "model_name": "qwen2.5vl:3b",
        "enabled": True,
        "base_url": "http://localhost:11434",
        "description": "Qwen2.5-VL 3B modeli",
        "token": None,
        "device": None
    },
    {
        "id": "deepseek_ocr",
        "platform": "huggingface",
        "model_name": "deepseek-ai/DeepSeek-OCR",
        "token": None,  # None ise environment variable'dan alınacak, veya token'ı buraya yazın
        "enabled": False,
        "description": "DeepSeek OCR Modeli"
    },
    {
        "id":"Qwen2.5-VL-7B-Instruct",
        "platform": "ollama",
        "model_name": "qwen2.5vl:7b",
        "enabled": True,
        "base_url": "http://localhost:11434",
        "description": "Qwen2.5-VL 7B modeli",
        "token": None,
        "device": None
    },
    {
        "id": "nanonets_ocr",
        "platform": "huggingface",
        "model_name": "nanonets/Nanonets-OCR2-3B",
        "token": None,  # None ise environment variable'dan alınacak, veya token'ı buraya yazın
        "enabled": True,
        "description": "Nanonets OCR Modeli"
    }
    
]

# Streamlit ayarları
STREAMLIT_CONFIG = {
    "page_title": "OCR Model Karşılaştırma",
    "page_icon": "🔍",
    "layout": "wide"
}

# Görüntü işleme ayarları
IMAGE_PROCESSING_CONFIG = {
    "max_image_size": (2048, 2048),
    "preprocess": {
        "enhance_contrast": False,
        "convert_grayscale": False,
        "resize": None
    }
}

# Model başlatma ayarları
MODEL_INIT_CONFIG = {
    "auto_initialize": False,  # Modelleri otomatik başlatma (kullanıcı seçimine göre)
    "initialize_on_startup": False  # Uygulama başlangıcında başlat
}

