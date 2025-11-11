"""
Ollama OCR Model Implementation
Ollama üzerindeki OCR modellerini kullanmak için implementasyon
"""

from typing import Dict, Optional
from PIL import Image
import base64
import io
import requests
import json
import os
import sys
from .base import BaseOCRModel, OCRResult

# Config modülünü import et (proje kök dizininden)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class OllamaOCR(BaseOCRModel):
    """Ollama platformundaki OCR modelleri için implementasyon"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        """
        Args:
            model_name: Ollama model adı (örn: 'qwen3-vl:2b')
            base_url: Ollama API base URL'i
        """
        super().__init__(model_name, "ollama")
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def initialize(self) -> bool:
        """Ollama modelinin kullanılabilir olduğunu kontrol et"""
        try:
            # Ollama'nın çalışıp çalışmadığını kontrol et
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Model mevcut mu kontrol et
                if self.model_name in model_names:
                    self.initialized = True
                    print(f"Ollama modeli hazır: {self.model_name}")
                    return True
                else:
                    print(f"Uyarı: Model '{self.model_name}' Ollama'da bulunamadı.")
                    print(f"Mevcut modeller: {model_names}")
                    # Yine de devam et, belki model yüklenebilir
                    self.initialized = True
                    return True
            else:
                print(f"Ollama API'ye erişilemiyor. Status: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"Ollama sunucusuna bağlanılamıyor: {self.base_url}")
            print("Ollama'nın çalıştığından emin olun: 'ollama serve'")
            return False
        except Exception as e:
            print(f"Ollama başlatma hatası: {str(e)}")
            return False
    
    def _image_to_base64(self, image: Image.Image, max_size: int = 512) -> str:
        """
        PIL Image'i base64 string'e çevir
        Büyük görüntüleri optimize eder (Ollama için önemli)
        """
        from PIL import ImageEnhance
        
        processed = image.copy()
        
        # RGBA veya diğer formatları RGB'ye çevir (JPEG için gerekli)
        if processed.mode in ('RGBA', 'LA', 'P'):
            # RGBA için beyaz arka plan oluştur
            if processed.mode == 'RGBA':
                background = Image.new('RGB', processed.size, (255, 255, 255))
                background.paste(processed, mask=processed.split()[3])  # Alpha channel'ı mask olarak kullan
                processed = background
            else:
                processed = processed.convert('RGB')
        elif processed.mode != 'RGB':
            processed = processed.convert('RGB')
        
        # Hafif kontrast artırma (çok fazla artırma küçük metinlere zarar verebilir)
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(1.1)  # Hafif kontrast artırma (%10)
        
        print("🎨 Görsel hafif kontrast artırıldı (OCR için)")
        
        # Çok büyük görüntüleri küçült (Ollama için performans - daha agresif)
        width, height = processed.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            processed = processed.resize(new_size, Image.Resampling.LANCZOS)
            print(f"📐 Görüntü boyutu optimize edildi: {width}x{height} -> {new_size[0]}x{new_size[1]}")
        
        # JPEG formatında kaydet (Ollama için optimize edilmiş - performans odaklı)
        buffered = io.BytesIO()
        # Kalite 70: Daha küçük dosya boyutu = daha hızlı işlem
        processed.save(buffered, format="JPEG", quality=70, optimize=True)
        file_size_kb = len(buffered.getvalue()) / 1024
        print(f"📦 Görüntü boyutu: {file_size_kb:.1f} KB")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64
    
    def predict(self, image: Image.Image) -> OCRResult:
        """OCR işlemi yap"""
        if not self.initialized:
            raise RuntimeError("Model başlatılmamış. Önce initialize() çağrılmalı.")
        
        try:
            # Görüntüyü ön işle
            processed_image = self.preprocess_image(image)
            
            # Görüntüyü base64'e çevir (Ollama için optimized - daha küçük boyut)
            # max_size=768: Performans ve kalite dengesi
            img_base64 = self._image_to_base64(processed_image, max_size=768)
            
            # Tüm modeller için ortak prompt (config.py'den alınır)
            # Türkçe metinler için optimize edilmiş, tutarlılık için tüm modellerde aynı
            prompt = config.OCR_PROMPT
            
            # Ollama API çağrısı - streaming ile daha hızlı yanıt
            # Streaming vs Normal Mod:
            # - Streaming: Token token yanıt gelir, daha hızlı başlangıç, kullanıcı deneyimi daha iyi
            # - Normal: Tüm yanıt bir kerede gelir, daha güvenilir ama daha yavaş başlangıç
            # Her ikisinde de aynı num_predict kullanılmalı (tutarlılık için)
            # num_predict=2048: Uzun metinler (CV gibi) için gerekli
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,  # Streaming açık - daha hızlı başlangıç
                "options": {
                    "temperature": 0.1,  # Hafif randomness - tam 0.0 yerine (daha iyi OCR için)
                    "top_p": 0.9,  # Nucleus sampling - daha tutarlı sonuçlar
                    "repeat_penalty": 1.2,  # Tekrarları azalt (1.0'dan arttırıldı)
                    "num_predict": 1536,  # Timeout önlemek için azaltıldı (2048'den)
                }
            }
            
            print(f"🔄 Ollama API çağrısı yapılıyor (streaming mode, timeout: 300s)...")
            
            # Streaming ile yanıt al
            # Streaming modunda timeout daha uzun olmalı çünkü token token geliyor
            # ve model yanıt üretmeye başlamadan önce timeout olmamalı
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300,  # Streaming için daha uzun timeout (5 dakika)
                stream=False
            )
            
            if response.status_code == 200:
                extracted_text = ""
                
                # Streaming yanıtını işle
                try:
                    chunk_count = 0
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                chunk_count += 1
                                
                                # Debug: İlk birkaç chunk'ı göster
                                if chunk_count <= 3:
                                    print(f"🔍 Chunk {chunk_count}: {list(chunk.keys())}")
                                
                                # Ollama streaming formatı - farklı alanları kontrol et
                                # Öncelik sırası: response > content > text > thinking
                                chunk_response = chunk.get("response", "")
                                chunk_content = chunk.get("content", "")
                                chunk_text = chunk.get("text", "")
                                chunk_thinking = chunk.get("thinking", "")
                                
                                # Response alanı varsa kullan (boş string değilse)
                                if chunk_response and chunk_response.strip():
                                    extracted_text += chunk_response
                                elif chunk_content and chunk_content.strip():
                                    extracted_text += chunk_content
                                elif chunk_text and chunk_text.strip():
                                    extracted_text += chunk_text
                                elif chunk_thinking and chunk_thinking.strip():
                                    # Thinking alanından metin çıkar (streaming modunda)
                                    # Thinking'i biriktir, son chunk'ta işle
                                    if not hasattr(self, '_thinking_buffer'):
                                        self._thinking_buffer = ""
                                    self._thinking_buffer += chunk_thinking
                                
                                # Done sinyali geldiğinde dur
                                if chunk.get("done", False):
                                    # Eğer response boşsa ve thinking buffer varsa, thinking'den çıkar
                                    if not extracted_text.strip() and hasattr(self, '_thinking_buffer') and self._thinking_buffer:
                                        thinking_extracted = self._extract_text_from_thinking(self._thinking_buffer)
                                        if thinking_extracted:
                                            extracted_text = thinking_extracted
                                            print("ℹ️  Metinler 'thinking' alanından çıkarıldı (streaming)")
                                        delattr(self, '_thinking_buffer')
                                    
                                    print(f"✅ Streaming tamamlandı. Toplam chunk: {chunk_count}, Toplam karakter: {len(extracted_text)}")
                                    break
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                print(f"⚠️  JSON decode hatası: {str(e)}, line: {line[:100] if line else 'None'}")
                                continue
                    
                    extracted_text = extracted_text.strip()
                    
                    # Streaming modunda prompt temizleme
                    if extracted_text:
                        prompt = config.OCR_PROMPT
                        if prompt in extracted_text:
                            extracted_text = extracted_text.replace(prompt, "").strip()
                        
                        # Prompt anahtar kelimelerini temizle
                        prompt_keywords = [
                            "Bu görüntüdeki tüm metinleri çıkar",
                            "Her kelimeyi, sayıyı ve karakteri",
                            "Satır sonlarını ve boşlukları koru",
                            "Türkçe karakterleri doğru şekilde kullan",
                            "Büyük/küçük harf ayrımına dikkat et",
                            "Noktalama işaretlerini ve özel karakterleri koru",
                            "Sadece çıkarılan metni döndür",
                            "Extract all text",
                            "Read every word",
                            "Preserve line breaks"
                        ]
                        
                        lines = extracted_text.split("\n")
                        filtered_lines = []
                        skip = False
                        for line in lines:
                            line_lower = line.lower().strip()
                            is_prompt_line = any(keyword.lower() in line_lower for keyword in prompt_keywords)
                            if is_prompt_line:
                                skip = True
                                continue
                            if skip and not line.strip():
                                skip = False
                                continue
                            if not skip:
                                filtered_lines.append(line)
                        extracted_text = "\n".join(filtered_lines).strip()
                    
                    # Debug: Eğer hala boşsa, son chunk'ı göster
                    if not extracted_text and chunk_count > 0:
                        print(f"⚠️  Streaming yanıtı boş. Toplam chunk sayısı: {chunk_count}")
                    
                    # Eğer streaming çalışmadıysa, normal yanıtı dene
                    if not extracted_text:
                        print("⚠️  Streaming yanıt alınamadı, normal mod deneniyor...")
                        payload["stream"] = False
                        retry_response = requests.post(
                            self.api_url,
                            json=payload,
                            timeout=300  # Normal mod için de uzun timeout
                        )
                        
                        if retry_response.status_code == 200:
                            result_data = retry_response.json()
                            
                            # Önce response alanını kontrol et
                            extracted_text = result_data.get("response", "").strip()
                            
                            # Eğer response boşsa, thinking alanını kontrol et
                            if not extracted_text and "thinking" in result_data:
                                thinking_text = result_data.get("thinking", "")
                                # Thinking içinden gerçek metinleri çıkar
                                extracted_text = self._extract_text_from_thinking(thinking_text)
                                if extracted_text:
                                    print("ℹ️  Metinler 'thinking' alanından çıkarıldı")
                            
                            # Eğer hala boşsa, diğer alanları kontrol et
                            if not extracted_text:
                                extracted_text = (
                                    result_data.get("text", "") or 
                                    result_data.get("content", "")
                                ).strip()
                        else:
                            print(f"❌ Normal mod hatası: {retry_response.status_code} - {retry_response.text}")
                
                except Exception as stream_error:
                    print(f"⚠️  Streaming işleme hatası: {str(stream_error)}")
                    # Fallback: Normal mod
                    try:
                        payload["stream"] = False
                        retry_response = requests.post(
                            self.api_url,
                            json=payload,
                            timeout=300  # Fallback için de uzun timeout
                        )
                        if retry_response.status_code == 200:
                            result_data = retry_response.json()
                            
                            # Önce response alanını kontrol et
                            extracted_text = result_data.get("response", "").strip()
                            
                            # Eğer response boşsa, thinking alanını kontrol et
                            if not extracted_text and "thinking" in result_data:
                                thinking_text = result_data.get("thinking", "")
                                extracted_text = self._extract_text_from_thinking(thinking_text)
                    except Exception as e:
                        print(f"❌ Fallback hatası: {str(e)}")
                
                # Prompt'u çıktıdan temizle (eğer varsa)
                if extracted_text:
                    prompt = config.OCR_PROMPT
                    # Prompt'un tamamını veya kısımlarını temizle
                    if prompt in extracted_text:
                        extracted_text = extracted_text.replace(prompt, "").strip()
                    
                    # Prompt'un anahtar kelimelerini temizle
                    prompt_keywords = [
                        "Bu görüntüdeki tüm metinleri çıkar",
                        "Her kelimeyi, sayıyı ve karakteri",
                        "Satır sonlarını ve boşlukları koru",
                        "Türkçe karakterleri doğru şekilde kullan",
                        "Büyük/küçük harf ayrımına dikkat et",
                        "Noktalama işaretlerini ve özel karakterleri koru",
                        "Sadece çıkarılan metni döndür",
                        "Extract all text",
                        "Read every word",
                        "Preserve line breaks"
                    ]
                    
                    lines = extracted_text.split("\n")
                    filtered_lines = []
                    skip = False
                    for line in lines:
                        line_lower = line.lower().strip()
                        # Prompt anahtar kelimelerini kontrol et
                        is_prompt_line = any(keyword.lower() in line_lower for keyword in prompt_keywords)
                        
                        if is_prompt_line:
                            skip = True
                            continue
                        if skip and not line.strip():
                            skip = False
                            continue
                        if not skip:
                            filtered_lines.append(line)
                    
                    extracted_text = "\n".join(filtered_lines).strip()
                
                # Debug: Yanıtı göster (sadece ilk 200 karakter)
                if extracted_text:
                    preview = extracted_text[:].replace('\n', ' ')
                    print(f"✅ Metin çıkarıldı ({len(extracted_text)} karakter): {preview}...")
                else:
                    print("⚠️  Model metin bulamadı veya yanıt vermedi.")
                    print("💡 İpuçları:")
                    print("   - Görüntüde gerçekten metin olduğundan emin olun")
                    print("   - Ollama modelinin çalıştığını kontrol edin: ollama list")
                    print("   - Farklı bir görüntü ile deneyin")
                
                # Ollama genelde sadece metin döndürür, bbox bilgisi yok
                return OCRResult(
                    text=extracted_text,
                    confidence=1.0 if extracted_text else 0.0,  # Ollama güven skoru döndürmez
                    bboxes=[],
                    words=[]
                )
            else:
                error_msg = f"Ollama API hatası: {response.status_code} - {response.text}"
                print(error_msg)
                return OCRResult(text="", confidence=0.0)
                
        except requests.exceptions.Timeout:
            print("❌ Ollama API çağrısı zaman aşımına uğradı (300 saniye)")
            print("💡 İpucu: Görüntü boyutunu küçültmeyi deneyin veya Ollama sunucusunun performansını kontrol edin")
            print(f"💡 Alternatif: num_predict değerini azaltmayı deneyin (şu an: {payload['options']['num_predict']})")
            print("💡 Önemli: Büyük modeller (4b, 8b) çok yavaş olabilir. Daha küçük model (2b) veya PaddleOCR deneyin")
            return OCRResult(text="", confidence=0.0)
        except requests.exceptions.ConnectionError:
            print("❌ Ollama sunucusuna bağlanılamadı")
            print("💡 İpucu: Ollama'nın çalıştığından emin olun: 'ollama serve'")
            return OCRResult(text="", confidence=0.0)
        except Exception as e:
            print(f"❌ Ollama OCR işlemi hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return OCRResult(text="", confidence=0.0)
    
    def _extract_text_from_thinking(self, thinking_text: str) -> str:
        """
        Thinking alanından sadece görüntüdeki gerçek metinleri çıkar
        Modelin kendi yorumlarını ve açıklamalarını filtreler
        """
        import re
        
        if not thinking_text:
            return ""
        
        # Thinking metninden sadece tırnak içindeki metinleri al
        quoted_texts = re.findall(r'"([^"]+)"', thinking_text)
        
        if not quoted_texts:
            print("⚠️  Thinking'de tırnak içinde metin bulunamadı")
            return ""
        
        print(f"🔍 Thinking'de {len(quoted_texts)} adet tırnaklı metin bulundu")
        
        # Tekrarları kaldır ve sadece gerçek metinleri filtrele
        unique_texts = []
        seen = set()
        
        # Model yorumlarını ve prompt metinlerini tespit etmek için filtreleme
        model_comment_indicators = [
            # Tamamen İngilizce model yorumları
            "let's", "let me", "i need to", "i should", "i'll", "i'm going to",
            "making sure", "preserve every", "the user wants", "tackle this",
            "including numbers", "special characters", "from the provided image",
            "the image shows", "looking at", "it appears", "i see that",
            "wait,", "maybe there", "starting with", "but let", "first,",
        ]
        
        # Prompt'un kendisini içeren metinler (bunlar kesinlikle filtrelenmeli)
        prompt_phrases = [
            "bu görüntüdeki",
            "tüm metinleri çıkar",
            "her kelimeyi",
            "satır sonlarını",
            "türkçe karakterleri doğru",
            "büyük/küçük harf",
            "noktalama işaretlerini",
            "sadece çıkarılan",
            "başka hiçbir",
            "açıklama veya yorum"
        ]
        
        for idx, text in enumerate(quoted_texts):
            text = text.strip()
            
            # Çok kısa metinleri atla (tek karakter veya boş)
            if len(text) < 2:
                continue
            
            text_lower = text.lower()
            
            # Prompt metinlerini filtrele (önce bunları kontrol et)
            is_prompt_text = any(phrase in text_lower for phrase in prompt_phrases)
            if is_prompt_text:
                print(f"  ✗ Metin #{idx+1} prompt metni (atlandı): {text[:50]}..." if len(text) > 50 else f"  ✗ Metin #{idx+1} prompt metni (atlandı): {text}")
                continue
            
            # Türkçe karakter içeriyorsa ve yeterince uzunsa, muhtemelen gerçek metindir
            has_turkish = any(char in text for char in 'ğüşöçİĞÜŞÖÇı')
            
            # Model yorumu kontrolü (sadece İngilizce metinler için)
            is_model_comment = False
            if not has_turkish:
                # İngilizce metin, model yorumu mu kontrol et
                for indicator in model_comment_indicators:
                    if indicator in text_lower:
                        is_model_comment = True
                        break
            
            # Eğer model yorumu değilse ve daha önce eklenmemişse ekle
            if not is_model_comment:
                text_normalized = text.lower().strip()
                if text_normalized not in seen:
                    seen.add(text_normalized)
                    unique_texts.append(text)
                    print(f"  ✓ Metin #{idx+1} eklendi: {text[:50]}..." if len(text) > 50 else f"  ✓ Metin #{idx+1} eklendi: {text}")
                else:
                    print(f"  ⊗ Metin #{idx+1} tekrar (atlandı): {text[:50]}..." if len(text) > 50 else f"  ⊗ Metin #{idx+1} tekrar (atlandı): {text}")
            else:
                print(f"  ✗ Metin #{idx+1} model yorumu (atlandı): {text[:50]}..." if len(text) > 50 else f"  ✗ Metin #{idx+1} model yorumu (atlandı): {text}")
        
        # Metinleri birleştir
        if unique_texts:
            result = "\n".join(unique_texts)
            # Fazla boşlukları temizle
            result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
            print(f"📊 Toplam {len(unique_texts)} metin birleştirildi, {len(result)} karakter")
            return result.strip()
        
        print("⚠️  Filtreleme sonrası hiç metin kalmadı")
        return ""
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            "model_name": self.model_name,
            "platform": self.platform,
            "base_url": self.base_url,
            "initialized": self.initialized
        }

