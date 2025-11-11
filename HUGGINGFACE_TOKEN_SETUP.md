# HuggingFace Token Kurulum Kılavuzu

Qwen3-VL-2B-Instruct gibi gated (korumalı) modelleri kullanmak için HuggingFace token gereklidir.

## 🔑 Token Alma

1. **HuggingFace hesabı oluşturun** (eğer yoksa):
   - https://huggingface.co/join adresinden kaydolun

2. **Token oluşturun**:
   - https://huggingface.co/settings/tokens adresine gidin
   - "New token" butonuna tıklayın
   - Token adı verin (örn: "ocr-models")
   - "Read" yetkisi yeterlidir
   - Token'ı kopyalayın (bir daha gösterilmeyecek!)

3. **Model erişimini onaylayın**:
   - Model sayfasına gidin: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
   - "Agree and access repository" butonuna tıklayın
   - Modeli kullanmak için onay verin

## 📝 Token Kullanım Yöntemleri

### Yöntem 1: Environment Variable (Önerilen)

**Windows (CMD):**
```cmd
set HF_TOKEN=hf_xxxxxxxxxxxxx
```

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxxx"
```

**Linux/Mac:**
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

**Kalıcı olarak eklemek (Windows):**
1. Sistem Özellikleri > Ortam Değişkenleri
2. Yeni kullanıcı değişkeni ekle
3. İsim: `HF_TOKEN`
4. Değer: Token'ınız

### Yöntem 2: Config Dosyası

`config.py` dosyasını açın ve token'ı ekleyin:

```python
HUGGINGFACE_TOKEN = "hf_xxxxxxxxxxxxx"
```

VEYA model yapılandırmasında:

```python
{
    "id": "qwen3-vl-2b",
    "platform": "huggingface",
    "model_name": "Qwen/Qwen3-VL-2B-Instruct",
    "token": "hf_xxxxxxxxxxxxx",  # Token buraya
    "enabled": True,
    "description": "Qwen3-VL 2B modeli (HuggingFace)"
}
```

### Yöntem 3: Streamlit Secrets (Güvenli)

`.streamlit/secrets.toml` dosyası oluşturun:

```toml
[secrets]
HF_TOKEN = "hf_xxxxxxxxxxxxx"
```

Sonra kodda kullanın:
```python
import streamlit as st
token = st.secrets.get("HF_TOKEN")
```

## ✅ Token Kontrolü

Token'ın doğru çalışıp çalışmadığını kontrol etmek için:

```python
import os
from huggingface_hub import whoami

token = os.getenv("HF_TOKEN")
if token:
    print(f"Token bulundu: {token[:10]}...")
    try:
        user_info = whoami(token=token)
        print(f"Kullanıcı: {user_info['name']}")
    except:
        print("Token geçersiz!")
else:
    print("Token bulunamadı!")
```

## 🚨 Hata Durumları

### 401 Unauthorized Hatası
- Token eksik veya geçersiz
- Model erişim izni verilmemiş
- Çözüm: Token'ı kontrol edin ve model sayfasından erişim izni verin

### Token Bulunamadı Uyarısı
- Environment variable ayarlanmamış
- Config dosyasında token yok
- Çözüm: Yukarıdaki yöntemlerden birini kullanın

## 📚 Daha Fazla Bilgi

- HuggingFace Token Dokümantasyonu: https://huggingface.co/docs/hub/security-tokens
- Model Erişim İzni: https://huggingface.co/docs/hub/models-access-gated

