"""
HuggingFace OCR Model Implementation
HuggingFace üzerindeki OCR modellerini kullanmak için implementasyon
"""

from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
import os
import sys
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq, 
    AutoModel,
    AutoModelForImageTextToText
)
from .base import BaseOCRModel, OCRResult

# Config modülünü import et (proje kök dizininden)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class HuggingFaceOCR(BaseOCRModel):
    """HuggingFace platformundaki OCR modelleri için implementasyon"""
    
    def __init__(
        self, 
        model_name: str, 
        model_path: Optional[str] = None, 
        device: Optional[str] = None,
        token: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model adı (örn: 'Qwen/Qwen3-VL-2B-Instruct')
            model_path: Alternatif olarak lokal model yolu
            device: Cihaz ('cuda', 'cpu', None=otomatik)
            token: HuggingFace token (gated modeller için gerekli)
        """
        super().__init__(model_name, "huggingface")
        self.model_path = model_path or model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Token'ı önce parametre, sonra environment variable'dan al
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.processor = None
        self.model = None
    
    def _patch_deepseek_flash_attention(self):
        """
        DeepSeek-OCR modelinin LlamaFlashAttention2 import hatasını düzeltir
        """
        import glob
        
        try:
            # Model cache klasörünü bul
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache = os.path.join(cache_dir, "models--deepseek-ai--DeepSeek-OCR")
            
            if not os.path.exists(model_cache):
                return
            
            # Snapshot klasörünü bul
            snapshots_dir = os.path.join(model_cache, "snapshots")
            if not os.path.exists(snapshots_dir):
                return
            
            # En son snapshot'ı bul
            snapshot_dirs = [d for d in glob.glob(os.path.join(snapshots_dir, "*")) if os.path.isdir(d)]
            if not snapshot_dirs:
                return
            
            latest_snapshot = max(snapshot_dirs, key=os.path.getmtime)
            modeling_file = os.path.join(latest_snapshot, "modeling_deepseekv2.py")
            
            if not os.path.exists(modeling_file):
                return
            
            # Dosyayı oku
            with open(modeling_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Eğer zaten patch edilmişse, tekrar yapma
            if "# Flash-attention opsiyonel" in content:
                print("✅ DeepSeek-OCR zaten patch edilmiş")
                return
            
            # LlamaFlashAttention2 import'unu bul ve değiştir
            old_import = """from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2
)"""
            
            new_import = """from transformers.models.llama.modeling_llama import LlamaAttention

# Flash-attention opsiyonel - varsa kullan, yoksa LlamaAttention kullan
try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
except ImportError:
    # LlamaFlashAttention2 mevcut değilse, LlamaAttention kullan
    LlamaFlashAttention2 = LlamaAttention
    print("⚠️  LlamaFlashAttention2 bulunamadı, LlamaAttention kullanılacak")"""
            
            if old_import in content:
                content = content.replace(old_import, new_import)
                
                # Dosyayı yaz
                with open(modeling_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print("✅ DeepSeek-OCR başarıyla patch edildi!")
            else:
                print("⚠️  DeepSeek-OCR patch gerekmiyor veya format farklı")
                
        except Exception as e:
            print(f"⚠️  DeepSeek-OCR patch hatası (göz ardı edildi): {str(e)}")
    
    def initialize(self) -> bool:
        """HuggingFace modelini yükle"""
        try:
            print(f"HuggingFace modeli yükleniyor: {self.model_path}")
            
            # Token kontrolü ve uyarı
            if not self.token:
                print("⚠️  Uyarı: HuggingFace token bulunamadı.")
                print("   Gated modeller için token gerekli olabilir.")
                print("   Token'ı şu şekillerde sağlayabilirsiniz:")
                print("   1. Environment variable: HF_TOKEN veya HUGGINGFACE_TOKEN")
                print("   2. Config dosyasında 'token' parametresi")
                print("   3. HuggingFace token almak için: https://huggingface.co/settings/tokens")
            
            # Processor ve model yükleme parametreleri
            load_kwargs = {
                "trust_remote_code": True
            }
            if self.token:
                load_kwargs["token"] = self.token
            
            # Processor yükleme - Qwen modelleri için özel işleme
            try:
                # Önce standart AutoProcessor ile dene
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    **load_kwargs
                )
            except Exception as proc_error:
                print(f"⚠️  AutoProcessor yükleme hatası: {str(proc_error)[:200]}")
                
                # Qwen modelleri için alternatif yöntemler
                model_name_lower = self.model_path.lower()
                if "qwen" in model_name_lower:
                    # Qwen2.5-VL için özel processor
                    if "qwen2.5" in model_name_lower or "qwen2_5" in model_name_lower:
                        try:
                            from transformers import Qwen2_5_VLProcessor
                            print("🔄 Qwen2_5_VLProcessor ile deneniyor...")
                            self.processor = Qwen2_5_VLProcessor.from_pretrained(
                                self.model_path,
                                **load_kwargs
                            )
                            print("✅ Qwen2_5_VLProcessor ile yüklendi")
                        except Exception as qwen25_error:
                            print(f"⚠️  Qwen2_5_VLProcessor başarısız: {str(qwen25_error)[:200]}")
                            # Qwen3-VL için Qwen2VLProcessor dene
                            try:
                                from transformers import Qwen2VLProcessor
                                print("🔄 Qwen2VLProcessor ile deneniyor...")
                                self.processor = Qwen2VLProcessor.from_pretrained(
                                    self.model_path,
                                    **load_kwargs
                                )
                                print("✅ Qwen2VLProcessor ile yüklendi")
                            except Exception as qwen_error:
                                print(f"⚠️  Qwen2VLProcessor başarısız: {str(qwen_error)[:200]}")
                                print("   Processor olmadan devam ediliyor...")
                                self.processor = None
                    else:
                        # Qwen3-VL veya diğer Qwen modelleri için
                        try:
                            from transformers import Qwen2VLProcessor
                            print("🔄 Qwen2VLProcessor ile deneniyor...")
                            self.processor = Qwen2VLProcessor.from_pretrained(
                                self.model_path,
                                **load_kwargs
                            )
                            print("✅ Qwen2VLProcessor ile yüklendi")
                        except Exception as qwen_error:
                            print(f"⚠️  Qwen2VLProcessor başarısız: {str(qwen_error)[:200]}")
                            print("   Processor olmadan devam ediliyor...")
                            self.processor = None
                else:
                    print("   Processor olmadan devam ediliyor...")
                    self.processor = None
            
            # Model yükleme - farklı model tiplerini desteklemek için esnek yükleme
            model_load_kwargs = {
                "trust_remote_code": True,
                "dtype": torch.float16 if self.device == "cuda" else torch.float32,  # torch_dtype yerine dtype
                "attn_implementation": "sdpa",  # Flash-attention yerine SDPA kullan (daha uyumlu)
                "local_files_only": False,  # Her zaman son versiyonu kullan
                "force_download": False  # Cache varsa kullan ama güncel olmayan dosyaları güncelle
            }
            if self.token:
                model_load_kwargs["token"] = self.token
            
            print(f"ℹ️  Attention implementation: sdpa (flash-attention gerektirmez)")
            print(f"ℹ️  Model yeni dosyalarla yüklenecek (cache temizlendi)")
            
            # DeepSeek-OCR için özel patch (LlamaFlashAttention2 hatası düzeltmesi)
            if "deepseek" in self.model_path.lower() and "ocr" in self.model_path.lower():
                self._patch_deepseek_flash_attention()
            
            # Önce AutoModelForImageTextToText ile dene (yeni API)
            try:
                print("🔄 AutoModelForImageTextToText ile yükleniyor...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    **model_load_kwargs
                )
                print("✅ AutoModelForImageTextToText ile yüklendi")
            except Exception as e1:
                # AutoModelForVision2Seq ile dene (eski API)
                try:
                    print(f"⚠️  AutoModelForImageTextToText başarısız: {str(e1)[:100]}")
                    print("🔄 AutoModelForVision2Seq ile yükleniyor...")
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.model_path,
                        **model_load_kwargs
                    )
                    print("✅ AutoModelForVision2Seq ile yüklendi")
                except Exception as e2:
                    # Son çare: AutoModel ile dene (en genel)
                    try:
                        print(f"⚠️  AutoModelForVision2Seq başarısız: {str(e2)[:100]}")
                        print("🔄 AutoModel ile yükleniyor (genel yükleme)...")
                        self.model = AutoModel.from_pretrained(
                            self.model_path,
                            **model_load_kwargs
                        )
                        print("✅ AutoModel ile yüklendi (özel model mimarisi)")
                    except Exception as e3:
                        raise Exception(f"Tüm model yükleme yöntemleri başarısız:\n"
                                      f"1. AutoModelForImageTextToText: {str(e1)[:200]}\n"
                                      f"2. AutoModelForVision2Seq: {str(e2)[:200]}\n"
                                      f"3. AutoModel: {str(e3)[:200]}")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.initialized = True
            print(f"✅ Model başarıyla yüklendi: {self.model_name}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Model yükleme hatası: {error_msg}")
            
            # Özel hata mesajları
            if "401" in error_msg or "Unauthorized" in error_msg:
                print("\n🔑 Bu model gated (korumalı) bir modeldir ve HuggingFace token gerektirir.")
                print("   Token almak için: https://huggingface.co/settings/tokens")
                print("   Token'ı şu şekillerde kullanabilirsiniz:")
                print("   1. Environment variable olarak: export HF_TOKEN='your_token'")
                print("   2. Config dosyasında 'token' parametresi ekleyin")
            
            self.initialized = False
            return False
    
    def predict(self, image: Image.Image) -> OCRResult:
        """OCR işlemi yap"""
        if not self.initialized:
            raise RuntimeError("Model başlatılmamış. Önce initialize() çağrılmalı.")
        
        try:
            # Model mimarisine göre otomatik tespit
            model_architecture = None
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type'):
                model_architecture = self.model.config.model_type
                print(f"🔍 Tespit edilen model mimarisi: {model_architecture}")
            
            # Model adını kontrol et
            model_name_lower = self.model_name.lower()
            
            # Qwen2.5-VL mimarisi (Nanonets-OCR2-3B da bu mimariye sahip)
            if (model_architecture == "qwen2_5_vl" or 
                "qwen2.5-vl" in model_name_lower or 
                "qwen2.5" in model_name_lower or
                "nanonets-ocr" in model_name_lower):
                print("📋 Qwen2.5-VL formatı kullanılıyor (chat template ile)")
                return self._predict_qwen25_vl(image)
            
            # Qwen3-VL mimarisi
            elif (model_architecture == "qwen3_vl" or
                  "qwen3-vl" in model_name_lower or 
                  "qwen3" in model_name_lower):
                print("📋 Qwen3-VL formatı kullanılıyor")
                return self._predict_qwen3_vl(image)
            
            # Görüntüyü ön işle
            processed_image = self.preprocess_image(image)
            
            # Processor kontrolü
            if self.processor is None:
                raise RuntimeError("Processor yüklenemedi. Model için processor gerekli.")
            
            # Tüm modeller için ortak prompt (config.py'den alınır)
            # Türkçe metinler için optimize edilmiş, tutarlılık için tüm modellerde aynı
            prompt = config.OCR_PROMPT
            
            # Model input hazırlama
            inputs = self.processor(
                images=processed_image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Inference - daha fazla token için
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=0.1
                )
                generated_text = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
            
            # Prompt'u çıktıdan temizle
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            # Sonuç oluşturma
            result = OCRResult(
                text=generated_text.strip(),
                confidence=1.0,  # Varsayılan güven skoru
                bboxes=[],
                words=[]
            )
            
            return result
            
        except Exception as e:
            print(f"OCR işlemi hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return OCRResult(text="", confidence=0.0)
    
    def _predict_qwen25_vl(self, image: Image.Image) -> OCRResult:
        """Qwen2.5-VL modelleri için özel OCR işlemi"""
        try:
            # Görüntüyü ön işle - daha az agresif (metinleri kaybetmemek için)
            processed_image = self._preprocess_for_ocr_light(image)
            
            # Tüm modeller için ortak prompt (config.py'den alınır)
            prompt_text = config.OCR_PROMPT
            
            # Processor kontrolü - Qwen2.5-VL için processor gerekli ama fallback var
            if self.processor is None:
                print("⚠️  Processor bulunamadı, Qwen2.5-VL için processor gerekli!")
                print("   Torchvision kurulumu gerekebilir: pip install torchvision")
                print("   Processor olmadan basit inference deneniyor...")
                
                # Processor olmadan basit inference dene
                try:
                    # Model'in kendi preprocess metodunu kullan
                    import torchvision.transforms as transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    inputs = transform(processed_image).unsqueeze(0).to(self.device)
                    
                    # Model inference
                    with torch.no_grad():
                        if hasattr(self.model, 'generate'):
                            # Tokenizer olmadan çalışamaz, hata ver
                            raise RuntimeError("Processor ve tokenizer gerekli")
                        else:
                            outputs = self.model(pixel_values=inputs)
                            # Output'u decode etmek için tokenizer gerekli
                            raise RuntimeError("Processor ve tokenizer gerekli")
                except Exception as fallback_error:
                    print(f"❌ Processor olmadan inference başarısız: {str(fallback_error)}")
                    print("💡 Çözüm: pip install torchvision komutunu çalıştırın")
                    return OCRResult(text="", confidence=0.0)
            
            # Qwen2.5-VL için özel chat template formatı kullan
            # Image token'larını prompt'a eklemek için özel format gerekli
            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"}, 
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            # Processor ile chat template'i apply et
            try:
                # Qwen2.5-VL processor'ı apply_chat_template ile kullanılmalı
                text_prompt = self.processor.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Image ve text'i birlikte işle
                inputs = self.processor(
                    images=[processed_image],
                    text=text_prompt,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
            except Exception as e:
                print(f"⚠️  Chat template hatası: {str(e)[:200]}")
                print("   Qwen2.5-VL için transformers 4.49.0 sürümü gerekebilir")
                print("   Komut: pip install transformers==4.49.0")
                raise e
            
            # Inference - Qwen2.5-VL için optimize edilmiş parametreler
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,  # Uzun metinler için yeterli token
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                )
                
                # Decode et
                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            
            # Chat template çıktısını temizle
            if "<|im_start|>" in generated_text:
                parts = generated_text.split("<|im_start|>")
                if len(parts) > 1:
                    assistant_part = parts[-1]
                    if "<|im_end|>" in assistant_part:
                        generated_text = assistant_part.split("<|im_end|>")[0]
                    else:
                        generated_text = assistant_part
                generated_text = generated_text.replace("assistant\n", "").strip()
            
            # Prompt'u çıktıdan temizle (detaylı temizleme)
            if prompt_text in generated_text:
                generated_text = generated_text.replace(prompt_text, "").strip()
            
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
                "Preserve line breaks",
                "Read all the text",
                "Extract every"
            ]
            
            lines = generated_text.split("\n")
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
            
            generated_text = "\n".join(filtered_lines).strip()
            
            return OCRResult(
                text=generated_text.strip(),
                confidence=1.0,
                bboxes=[],
                words=[]
            )
            
        except Exception as e:
            print(f"Qwen2.5-VL OCR işlemi hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return OCRResult(text="", confidence=0.0)
    
    def _predict_qwen3_vl(self, image: Image.Image) -> OCRResult:
        """Qwen3-VL modelleri için özel OCR işlemi"""
        try:
            # Görüntüyü ön işle - daha az agresif (metinleri kaybetmemek için)
            processed_image = self._preprocess_for_ocr_light(image)
            
            # Tüm modeller için ortak prompt (config.py'den alınır)
            # Türkçe metinler için optimize edilmiş, tutarlılık için tüm modellerde aynı
            prompt_text = config.OCR_PROMPT
            
            # Qwen3-VL için chat template formatı
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": processed_image,
                        },
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                    ],
                }
            ]
            
            # Chat template ile input hazırlama
            try:
                # Qwen3-VL için chat template formatı
                # Processor'ın apply_chat_template metodunu kullan
                if hasattr(self.processor, 'apply_chat_template'):
                    text = self.processor.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    # Processor ile görüntü ve metni birlikte işle
                    # Qwen3-VL processor'ı genelde doğrudan messages formatını kabul eder
                    inputs = self.processor(
                        messages,
                        return_tensors="pt"
                    ).to(self.device)
                else:
                    # Eğer apply_chat_template yoksa, manuel format
                    raise AttributeError("apply_chat_template not found")
                    
            except (AttributeError, TypeError, Exception) as e:
                # Fallback: Basit format kullan
                print(f"⚠️  Chat template kullanılamıyor ({str(e)}), basit format deneniyor...")
                # Tüm modeller için ortak prompt (config.py'den alınır)
                prompt = config.OCR_PROMPT
                inputs = self.processor(
                    images=processed_image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
            
            # Inference - Qwen3-VL için optimize edilmiş parametreler
            # Daha fazla token ve daha iyi sampling parametreleri
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8192,  # Artırıldı - çok uzun metinler için (sadece 1 satır sorunu için)
                    do_sample=True,  # Sampling açık - daha iyi sonuçlar için
                    temperature=0.3,  # Biraz daha yüksek - daha çeşitli çıktılar
                    top_p=0.95,  # Nucleus sampling
                    repetition_penalty=1.15,  # Tekrarları azalt
                    pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                )
                
                # Decode et
                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            
            # Chat template çıktısını temizle
            # Qwen3-VL genelde <|im_start|> ve <|im_end|> tokenları kullanır
            if "<|im_start|>" in generated_text:
                parts = generated_text.split("<|im_start|>")
                if len(parts) > 1:
                    assistant_part = parts[-1]
                    if "<|im_end|>" in assistant_part:
                        generated_text = assistant_part.split("<|im_end|>")[0]
                    else:
                        generated_text = assistant_part
                generated_text = generated_text.replace("assistant\n", "").strip()
            
            # Prompt'u çıktıdan temizle (detaylı temizleme)
            if prompt_text in generated_text:
                generated_text = generated_text.replace(prompt_text, "").strip()
            
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
                "Preserve line breaks",
                "Read all the text",
                "Extract every"
            ]
            
            lines = generated_text.split("\n")
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
            
            generated_text = "\n".join(filtered_lines).strip()
            
            result = OCRResult(
                text=generated_text.strip(),
                confidence=1.0,
                bboxes=[],
                words=[]
            )
            
            return result
            
        except Exception as e:
            print(f"Qwen3-VL OCR işlemi hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback: Basit yöntemi dene
            try:
                processed_image = self._preprocess_for_ocr_light(image)
                # Tüm modeller için ortak prompt (config.py'den alınır)
                prompt = config.OCR_PROMPT
                inputs = self.processor(
                    images=processed_image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=4096,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.95,
                        repetition_penalty=1.15
                    )
                    generated_text = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]
                
                return OCRResult(text=generated_text.strip(), confidence=0.8)
            except Exception as fallback_error:
                print(f"Fallback yöntem hatası: {str(fallback_error)}")
                return OCRResult(text="", confidence=0.0)
    
    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """OCR için görüntü ön işleme - CV/resume için optimize edilmiş"""
        from PIL import ImageEnhance
        
        processed = image.copy()
        
        # RGB formatına çevir (gerekirse)
        if processed.mode != 'RGB':
            processed = processed.convert('RGB')
        
        # Çok büyük görüntüleri optimize et (max 2048px)
        max_size = 2048
        width, height = processed.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            processed = processed.resize(new_size, Image.Resampling.LANCZOS)
        
        # Kontrast ve keskinliği hafifçe artır (CV'ler için önemli)
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(1.2)  # %20 kontrast artışı
        
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(1.1)  # %10 keskinlik artışı
        
        return processed
    
    def _preprocess_for_ocr_light(self, image: Image.Image) -> Image.Image:
        """OCR için hafif görüntü ön işleme - metinleri kaybetmemek için minimal işleme"""
        processed = image.copy()
        
        # RGB formatına çevir (gerekirse)
        if processed.mode != 'RGB':
            processed = processed.convert('RGB')
        
        # Çok büyük görüntüleri optimize et (max 2048px) - ama daha yüksek kalite için
        max_size = 2048
        width, height = processed.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            # LANCZOS daha iyi kalite sağlar
            processed = processed.resize(new_size, Image.Resampling.LANCZOS)
        
        # Kontrast ve keskinlik artırımı yapmıyoruz - orijinal görüntüyü koruyoruz
        # Çünkü agresif işleme metinleri bozabilir
        
        return processed
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            "model_name": self.model_name,
            "platform": self.platform,
            "model_path": self.model_path,
            "device": self.device,
            "has_token": self.token is not None,
            "initialized": self.initialized
        }


class PaddleOCRWrapper(BaseOCRModel):
    """PaddleOCR için özel wrapper (HuggingFace benzeri yapı)"""
    
    def __init__(self, model_name: str = "paddleocr"):
        super().__init__(model_name, "paddleocr")
        self.paddleocr = None
    
    def initialize(self) -> bool:
        """PaddleOCR'ı yükle"""
        try:
            # Önce paddlepaddle'ı kontrol et
            try:
                import paddle
                print(f"✅ PaddlePaddle bulundu: {paddle.__version__}")
            except ImportError:
                print("❌ PaddlePaddle yüklü değil.")
                print("   PaddleOCR için önce PaddlePaddle yüklenmelidir:")
                print("   pip install paddlepaddle")
                print("   veya en son versiyon için: pip install paddlepaddle>=3.0.0")
                return False
            
            from paddleocr import PaddleOCR
            
            print(f"PaddleOCR modeli yükleniyor...")
            # OCR engine başlat (use_angle_cls=True, lang='en' veya 'tr')
            # Türkçe için lang='tr' kullanabilirsiniz
            # Not: Yeni PaddleOCR versiyonlarında show_log parametresi desteklenmiyor
            self.paddleocr = PaddleOCR(use_angle_cls=True, lang='tr')
            self.initialized = True
            print(f"✅ PaddleOCR başarıyla yüklendi")
            return True
            
        except ImportError as e:
            print(f"❌ PaddleOCR import hatası: {str(e)}")
            print("   Kurulum için:")
            print("   1. pip install paddlepaddle")
            print("   2. pip install paddleocr")
            print("   Not: En son versiyonlar için:")
            print("   pip install paddlepaddle>=3.0.0")
            print("   pip install paddleocr>=2.7.0")
            return False
        except Exception as e:
            print(f"❌ PaddleOCR yükleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, image: Image.Image) -> OCRResult:
        """OCR işlemi yap"""
        if not self.initialized:
            raise RuntimeError("Model başlatılmamış.")
        
        try:
            import numpy as np
            
            # PIL Image'ı RGB formatına dönüştür (RGBA, LA, P gibi formatlar sorun çıkarabilir)
            original_mode = image.mode
            if image.mode != 'RGB':
                print(f"⚠️ Görüntü formatı {original_mode}, RGB'ye dönüştürülüyor...")
                # RGBA veya LA modunda ise beyaz arka plan üzerine çiz
                if image.mode in ('RGBA', 'LA', 'P'):
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'P':
                        image = image.convert('RGBA')
                    rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                    image = rgb_image
                else:
                    image = image.convert('RGB')
            
            # PIL Image'i numpy array'e çevir
            img_array = np.array(image)
            
            # Debug: Görüntü formatını kontrol et
            print(f"✅ Görüntü formatı: {img_array.shape}, dtype: {img_array.dtype}")
            
            # PaddleOCR ile işle
            # Not: cls parametresi yeni versiyonlarda desteklenmiyor
            # use_angle_cls zaten constructor'da belirtildi
            result = self.paddleocr.ocr(img_array)
            
            # Debug: Sonuç formatını kontrol et
            print(f"🔍 PaddleOCR sonuç tipi: {type(result)}")
            
            if result is None:
                print("⚠️ PaddleOCR sonuç bulunamadı (None)")
                return OCRResult(text="", confidence=0.0)
            
            # Sonuçları parse et
            text_parts = []
            bboxes = []
            words = []
            
            # Yeni PaddleOCR versiyonu dictionary formatında sonuç döndürüyor
            # Format: [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...], 'rec_boxes': [...]}, ...]
            if isinstance(result, list) and len(result) > 0:
                # İlk sonucu al (genellikle tek bir sayfa için bir sonuç döner)
                ocr_result = result[0]
                
                # Dictionary formatını kontrol et
                if isinstance(ocr_result, dict):
                    rec_texts = ocr_result.get('rec_texts', [])
                    rec_scores = ocr_result.get('rec_scores', [])
                    rec_polys = ocr_result.get('rec_polys', [])
                    rec_boxes = ocr_result.get('rec_boxes', None)
                    
                    print(f"✅ Yeni format tespit edildi: {len(rec_texts)} metin bulundu")
                    
                    # Her metin için işle
                    for idx, text in enumerate(rec_texts):
                        if not text or not isinstance(text, str):
                            continue
                        
                        text = text.strip()
                        if not text:
                            continue
                        
                        # Güven skorunu al
                        confidence = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
                        
                        # Bounding box'ı al
                        bbox_rect = None
                        
                        # rec_polys'i kullan (en güvenilir format)
                        if idx < len(rec_polys):
                            try:
                                import numpy as np
                                poly = rec_polys[idx]
                                if isinstance(poly, np.ndarray) and poly.shape[0] >= 4:
                                    # Polygon formatı: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                    x_coords = [int(point[0]) for point in poly]
                                    y_coords = [int(point[1]) for point in poly]
                                    bbox_rect = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                            except Exception as e:
                                print(f"⚠️ Bbox parse hatası (rec_polys): {e}")
                        
                        # Metni ekle
                        text_parts.append(text)
                        
                        if bbox_rect:
                            bboxes.append(bbox_rect)
                            words.append({
                                "text": text,
                                "bbox": bbox_rect,
                                "confidence": confidence
                            })
                        else:
                            # Bbox yoksa sadece metni ekle
                            words.append({
                                "text": text,
                                "bbox": None,
                                "confidence": confidence
                            })
                
                # Eski format kontrolü (geriye dönük uyumluluk için)
                elif isinstance(ocr_result, list):
                    print("⚠️ Eski format tespit edildi, parse ediliyor...")
                    # Eski format parse mantığı buraya eklenebilir
                    for line in ocr_result:
                        if isinstance(line, list) and len(line) >= 2:
                            try:
                                bbox = line[0]
                                text_info = line[1]
                                
                                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                    text = text_info[0]
                                    confidence = float(text_info[1]) if len(text_info) > 1 else 0.0
                                else:
                                    text = str(text_info)
                                    confidence = 0.0
                                
                                if text:
                                    text_parts.append(text)
                                    
                                    if isinstance(bbox, list) and len(bbox) >= 4:
                                        x_coords = [point[0] for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]
                                        y_coords = [point[1] for point in bbox if isinstance(point, (list, tuple)) and len(point) >= 2]
                                        
                                        if x_coords and y_coords:
                                            bbox_rect = (int(min(x_coords)), int(min(y_coords)), 
                                                       int(max(x_coords)), int(max(y_coords)))
                                            bboxes.append(bbox_rect)
                                            words.append({
                                                "text": text,
                                                "bbox": bbox_rect,
                                                "confidence": confidence
                                            })
                            except Exception as e:
                                print(f"⚠️ Eski format parse hatası: {e}")
                                continue
            
            print(f"✅ Parse edilen metin sayısı: {len(text_parts)}")
            
            full_text = "\n".join(text_parts)
            avg_confidence = sum([w["confidence"] for w in words]) / len(words) if words else 0.0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                bboxes=bboxes,
                words=words
            )
            
        except Exception as e:
            import traceback
            print(f"PaddleOCR işlemi hatası: {str(e)}")
            print(f"Hata detayı: {traceback.format_exc()}")
            return OCRResult(text="", confidence=0.0)
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            "model_name": self.model_name,
            "platform": self.platform,
            "initialized": self.initialized
        }

