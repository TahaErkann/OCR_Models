# 📄 PDF OCR Kullanım Kılavuzu

Bu proje artık **PDF belgelerini** OCR ile işleyebilir!

## 🎯 Özellikler

### ✅ PDF Desteği
- **PDF dosyalarını doğrudan yükleyin** - Sayfa sayfa OCR işlemi
- **Tüm sayfaları veya belirli aralıkları işleyin** - Esnek sayfa seçimi
- **Yüksek kaliteli görüntü dönüşümü** - Ayarlanabilir DPI (150-600)
- **Sayfa bazında sonuçlar** - Her sayfa için ayrı OCR sonucu
- **Tam metin çıktısı** - Tüm sayfalar birleştirilmiş metin
- **Metin indirme** - `.txt` formatında kaydetme

### 🔧 Teknik Detaylar
- **İki PDF backend** - `pdf2image` (önerilen) veya `PyMuPDF` (alternatif)
- **Otomatik backend seçimi** - Mevcut kütüphaneyi otomatik tespit eder
- **Progress tracking** - Her sayfanın işlem durumunu gösterir
- **Hata yönetimi** - Detaylı hata mesajları ve çözüm önerileri

## 📦 Kurulum

### 1. PDF Kütüphanelerini Kurun

```bash
pip install pdf2image PyMuPDF
```

### 2. Poppler Kurulumu (pdf2image için)

**Windows:**
1. [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/) indirin
2. Zip dosyasını çıkarın (örn: `C:\poppler`)
3. `C:\poppler\Library\bin` klasörünü PATH'e ekleyin

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Not:** `PyMuPDF` kullanıyorsanız Poppler gerekmez!

## 🚀 Kullanım

### 1. Streamlit Başlatma

```bash
streamlit run app.py
```

### 2. PDF OCR Tab'ına Gidin

Arayüzde **"📄 PDF OCR"** tab'ına tıklayın.

### 3. Model Seçin

Sidebar'dan bir OCR modeli seçin ve başlatın:
- **PaddleOCR** - En hızlı ve güvenilir (önerilen)
- **Qwen3-VL** - Vision-language model
- **DeepSeek-OCR** - Gelişmiş OCR modeli

### 4. PDF Yükleyin

- **"PDF Dosyası Yükleyin"** butonuna tıklayın
- PDF dosyanızı seçin
- Sayfa sayısı otomatik gösterilir

### 5. İşlem Ayarları

**Sayfa Seçimi:**
- ✅ **Tüm Sayfaları İşle** - PDF'in tüm sayfalarını işle
- ⬜ **Belirli Aralık** - Başlangıç ve bitiş sayfası seçin

**Gelişmiş Ayarlar:**
- **DPI (150-600)** - Yüksek DPI = Daha iyi kalite ama daha yavaş
  - 150 DPI: Hızlı ama düşük kalite
  - 300 DPI: Dengeli (önerilen)
  - 600 DPI: En yüksek kalite ama çok yavaş

### 6. OCR İşlemi

1. **"🔍 PDF'i OCR İle İşle"** butonuna tıklayın
2. Progress bar ile ilerlemeyi takip edin
3. Sonuçları inceleyin:
   - **Tam Metin** - Tüm sayfalar birleştirilmiş
   - **Sayfa Detayları** - Her sayfa için ayrı sonuç

### 7. Sonuçları İndirin

- **"💾 Metni İndir (.txt)"** butonuna tıklayın
- Tüm metin `.txt` dosyası olarak kaydedilir

## 📊 Örnek Kullanım Senaryoları

### Senaryo 1: Taranmış Belge OCR
```
1. Taranmış PDF belgenizi yükleyin
2. PaddleOCR modelini seçin (hızlı ve güvenilir)
3. DPI = 300 (dengeli kalite)
4. "Tüm Sayfaları İşle" seçeneğini işaretleyin
5. OCR işlemini başlatın
6. Sonuçları .txt olarak indirin
```

### Senaryo 2: Çok Sayfalı Kitap/Makale
```
1. PDF'inizi yükleyin
2. İlk 10 sayfayı test etmek için:
   - "Tüm Sayfaları İşle"yi kaldırın
   - Başlangıç: 1, Bitiş: 10
3. Sonuç tatmin ediciyse tüm PDF için tekrarlayın
```

### Senaryo 3: Düşük Kaliteli PDF
```
1. PDF'inizi yükleyin
2. DPI'yi 600'e çıkarın (yüksek kalite)
3. DeepSeek-OCR veya Qwen3-VL kullanın (daha gelişmiş)
4. İşlem daha uzun sürer ama daha iyi sonuç verir
```

## 🔍 Backend Bilgileri

### pdf2image (Önerilen)
- ✅ En yüksek kalite
- ✅ Poppler tabanlı (endüstri standardı)
- ⚠️ Poppler kurulumu gerektirir

### PyMuPDF (Alternatif)
- ✅ Ekstra kurulum gerektirmez
- ✅ Daha hızlı
- ⚠️ Biraz daha düşük kalite

## ⚙️ Config Entegrasyonu

Tüm mevcut OCR modelleri PDF ile uyumludur:

```python
# config.py
MODEL_CONFIGS = [
    {
        "id": "paddleocr",
        "platform": "paddleocr",
        "model_name": "PaddleOCR",
        "enabled": True
    },
    {
        "id": "qwen3_vl_2b",
        "platform": "ollama",
        "model_name": "qwen3-vl:2b",
        "enabled": True
    },
    # ... diğer modeller
]
```

## 🐛 Sorun Giderme

### "PDF backend bulunamadı" Hatası
```bash
pip install pdf2image PyMuPDF
```

### "Poppler bulunamadı" Hatası
- Windows: Poppler'ı indirin ve PATH'e ekleyin
- Linux: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`
- **Veya:** `PyMuPDF` kullanın (Poppler gerektirmez)

### OCR Sonuçları Kötü
1. DPI'yi artırın (300 → 600)
2. Daha güçlü bir model seçin (PaddleOCR → DeepSeek-OCR)
3. PDF'in kalitesini kontrol edin (taranmış mı, dijital mi?)

### Yavaş İşlem
1. DPI'yi düşürün (300 → 150)
2. Daha az sayfa işleyin
3. Daha hızlı bir model kullanın (PaddleOCR)

## 🎯 En İyi Pratikler

1. **İlk test için az sayfa kullanın** - Ayarları optimize edin
2. **DPI = 300** - Çoğu durumda yeterli
3. **PaddleOCR kullanın** - En hızlı ve güvenilir
4. **Sonuçları kaydedin** - `.txt` formatında indirin
5. **Sayfa aralıklarıyla çalışın** - Çok büyük PDF'lerde

## 📝 Notlar

- PDF işleme **orijinal projeyi etkilemez** - Ayrı bir tab
- Tüm mevcut modeller **PDF ile uyumludur**
- **Model ekleme/çıkarma** - `config.py` üzerinden yapılır
- PDF sayfaları **geçici olarak RAM'de tutulur** - Çok büyük PDF'lerde dikkat

