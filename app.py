"""
Streamlit OCR Model Comparison App
OCR modellerini karşılaştırmak için Streamlit arayüzü
"""

import streamlit as st
from PIL import Image
import io
import sys
import os

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.registry import ModelRegistry
from utils.image_processor import ImageProcessor
from utils.comparison import ModelComparator
from utils.pdf_processor import PDFProcessor
import config


# Sayfa yapılandırması
st.set_page_config(
    page_title=config.STREAMLIT_CONFIG["page_title"],
    page_icon=config.STREAMLIT_CONFIG["page_icon"],
    layout=config.STREAMLIT_CONFIG["layout"]
)

# Session state başlatma
if 'registry' not in st.session_state:
    st.session_state.registry = ModelRegistry()
    st.session_state.models_initialized = False
    st.session_state.uploaded_image = None
    st.session_state.comparison_results = None
    st.session_state.pdf_processor = PDFProcessor(dpi=300)
    st.session_state.pdf_pages = None
    st.session_state.current_pdf = None

# Başlık
st.title("🔍 OCR Model Karşılaştırma Sistemi")
st.markdown("---")

# Sidebar - Model Yönetimi
with st.sidebar:
    st.header("📋 Model Yönetimi")
    
    # Config'den modelleri yükle
    if not st.session_state.models_initialized:
        st.info("Modeller yükleniyor...")
        registry = st.session_state.registry
        
        for model_config in config.MODEL_CONFIGS:
            if model_config.get("enabled", True):
                platform = model_config["platform"]
                model_name = model_config["model_name"]
                model_id = model_config["id"]
                
                kwargs = {}
                if platform == "ollama":
                    kwargs["base_url"] = model_config.get("base_url", "http://localhost:11434")
                elif platform == "huggingface":
                    kwargs["device"] = model_config.get("device", None)
                    # Token'ı config'den veya global config'den al
                    token = model_config.get("token") or config.HUGGINGFACE_TOKEN
                    if token:
                        kwargs["token"] = token
                
                registry.register_model(
                    model_id=model_id,
                    platform=platform,
                    model_name=model_name,
                    **kwargs
                )
        
        st.session_state.models_initialized = True
        st.success("Modeller yüklendi!")
    
    # Model listesi
    st.subheader("Kayıtlı Modeller")
    registry = st.session_state.registry
    model_ids = registry.get_model_ids()
    
    if not model_ids:
        st.warning("Henüz model kaydedilmemiş!")
    else:
        for model_id in model_ids:
            model = registry.get_model(model_id)
            if model:
                info = model.get_model_info()
                initialized = "✅" if model.is_initialized() else "❌"
                st.write(f"{initialized} **{model_id}**")
                st.caption(f"Platform: {info.get('platform', 'N/A')}")
                
                # Model başlatma butonu
                if not model.is_initialized():
                    if st.button(f"Başlat", key=f"init_{model_id}"):
                        with st.spinner(f"{model_id} başlatılıyor..."):
                            success = model.initialize()
                            if success:
                                st.success(f"{model_id} başlatıldı!")
                                st.rerun()
                            else:
                                st.error(f"{model_id} başlatılamadı!")
    
    st.markdown("---")
    
    # Yeni model ekleme (gelişmiş)
    with st.expander("➕ Yeni Model Ekle"):
        st.info("Yeni modeller için config.py dosyasını düzenleyin.")

# Ana içerik
tab1, tab2, tab3, tab4 = st.tabs(["📸 Image OCR", "📄 PDF OCR", "📊 Karşılaştırma", "ℹ️ Bilgi"])

# TAB 1: OCR Testi
with tab1:
    st.header("Tek Model Testi")
    
    # Model seçimi
    if model_ids:
        selected_model_id = st.selectbox(
            "Model Seçin:",
            options=model_ids,
            key="single_model_select"
        )
        
        selected_model = registry.get_model(selected_model_id)
        
        if selected_model and selected_model.is_initialized():
            # Görüntü yükleme
            uploaded_file = st.file_uploader(
                "Görüntü yükleyin:",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key="single_image_upload"
            )
            
            if uploaded_file is not None:
                # Görüntüyü yükle
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                
                # Görüntüyü göster
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Orijinal Görüntü")
                    st.image(image, width='stretch')
                
                # OCR işlemi
                if st.button("🔍 OCR İşlemi Yap", key="single_ocr_button"):
                    with st.spinner("OCR işlemi yapılıyor..."):
                        try:
                            result = selected_model.predict(image)
                            
                            with col2:
                                st.subheader("OCR Sonuçları")
                                
                                # Metin sonucu
                                st.text_area(
                                    "Tespit Edilen Metin:",
                                    value=result.text,
                                    height=300,
                                    key="single_result_text"
                                )
                                
                                # Güven skoru
                                st.metric("Güven Skoru", f"{result.confidence:.2%}")
                                
                                # Bounding box görselleştirme
                                if result.bboxes:
                                    processor = ImageProcessor()
                                    visualized = processor.draw_bounding_boxes(
                                        image,
                                        result.bboxes,
                                        labels=[f"Text {i+1}" for i in range(len(result.bboxes))]
                                    )
                                    st.image(visualized, width='stretch', caption="Tespit Edilen Metinler")
                                
                                # Detaylı bilgiler
                                if result.words:
                                    with st.expander("Detaylı Kelime Bilgileri"):
                                        for idx, word in enumerate(result.words):
                                            st.write(f"**Kelime {idx+1}:** {word.get('text', 'N/A')}")
                                            st.write(f"Güven: {word.get('confidence', 0):.2%}")
                                            
                        except Exception as e:
                            st.error(f"Hata: {str(e)}")
        else:
            st.warning(f"Seçilen model başlatılmamış. Lütfen sidebar'dan modeli başlatın.")
    else:
        st.warning("Henüz model kaydedilmemiş!")

# TAB 2: PDF OCR
with tab2:
    st.header("PDF Belgesi OCR")
    st.info("📄 PDF dosyalarını yükleyip sayfa sayfa OCR işlemi yapabilirsiniz.")
    
    # Model seçimi
    if model_ids:
        selected_pdf_model_id = st.selectbox(
            "PDF için Model Seçin:",
            options=model_ids,
            key="pdf_model_select"
        )
        
        selected_pdf_model = registry.get_model(selected_pdf_model_id)
        
        if selected_pdf_model and selected_pdf_model.is_initialized():
            # PDF yükleme
            uploaded_pdf = st.file_uploader(
                "PDF Dosyası Yükleyin:",
                type=['pdf'],
                key="pdf_upload"
            )
            
            if uploaded_pdf is not None:
                try:
                    # PDF bilgilerini göster
                    pdf_processor = st.session_state.pdf_processor
                    
                    with st.spinner("PDF analiz ediliyor..."):
                        # Sayfa sayısını al
                        page_count = pdf_processor.get_page_count(uploaded_pdf)
                        st.success(f"✅ PDF yüklendi: {page_count} sayfa")
                        
                        # PDF'i session'a kaydet
                        if uploaded_pdf.name != st.session_state.current_pdf:
                            st.session_state.current_pdf = uploaded_pdf.name
                            st.session_state.pdf_pages = None
                    
                    # İşlem seçenekleri
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        process_all = st.checkbox("Tüm Sayfaları İşle", value=True, key="pdf_process_all")
                    
                    with col2:
                        if not process_all:
                            start_page = st.number_input(
                                "Başlangıç Sayfası:",
                                min_value=1,
                                max_value=page_count,
                                value=1,
                                key="pdf_start_page"
                            )
                    
                    with col3:
                        if not process_all:
                            end_page = st.number_input(
                                "Bitiş Sayfası:",
                                min_value=start_page if not process_all else 1,
                                max_value=page_count,
                                value=min(start_page + 2, page_count) if not process_all else page_count,
                                key="pdf_end_page"
                            )
                    
                    # DPI ayarı
                    with st.expander("⚙️ Gelişmiş Ayarlar"):
                        dpi = st.slider(
                            "PDF Görüntü Kalitesi (DPI):",
                            min_value=150,
                            max_value=600,
                            value=300,
                            step=50,
                            help="Yüksek DPI daha iyi kalite ama daha yavaş işlem"
                        )
                        pdf_processor.dpi = dpi
                    
                    # OCR işlemi butonu
                    if st.button("🔍 PDF'i OCR İle İşle", key="pdf_ocr_button"):
                        with st.spinner("PDF sayfaları görüntülere dönüştürülüyor..."):
                            try:
                                # Sayfa aralığını belirle
                                first_page = 1 if process_all else start_page
                                last_page = None if process_all else end_page
                                
                                # PDF'i görüntülere dönüştür
                                uploaded_pdf.seek(0)  # Dosya pointer'ını başa al
                                pages = pdf_processor.pdf_to_images(
                                    uploaded_pdf,
                                    first_page=first_page,
                                    last_page=last_page
                                )
                                
                                st.success(f"✅ {len(pages)} sayfa görüntüye dönüştürüldü")
                                
                                # Her sayfayı OCR ile işle
                                st.subheader("📄 Sayfa Bazında OCR Sonuçları")
                                
                                # Progress bar
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                all_results = []
                                
                                for idx, page_image in enumerate(pages):
                                    page_num = (first_page if first_page else 1) + idx
                                    
                                    # Progress güncelle
                                    progress = (idx + 1) / len(pages)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Sayfa {page_num} işleniyor... ({idx + 1}/{len(pages)})")
                                    
                                    # OCR işlemi
                                    result = selected_pdf_model.predict(page_image)
                                    
                                    # Sonucu kaydet
                                    all_results.append({
                                        'page': page_num,
                                        'text': result.text,
                                        'confidence': result.confidence,
                                        'image': page_image
                                    })
                                
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Sonuçları göster
                                st.success(f"✅ {len(pages)} sayfa başarıyla işlendi!")
                                
                                # Tam metin (tüm sayfalar birleştirilmiş)
                                with st.expander("📝 Tam Metin (Tüm Sayfalar)", expanded=False):
                                    full_text = "\n\n".join([
                                        f"--- Sayfa {r['page']} ---\n{r['text']}"
                                        for r in all_results
                                    ])
                                    st.text_area(
                                        "Birleştirilmiş Metin:",
                                        value=full_text,
                                        height=400,
                                        key="pdf_full_text"
                                    )
                                    
                                    # İndirme butonu
                                    st.download_button(
                                        label="💾 Metni İndir (.txt)",
                                        data=full_text,
                                        file_name=f"{uploaded_pdf.name.replace('.pdf', '')}_ocr.txt",
                                        mime="text/plain"
                                    )
                                
                                # Sayfa bazında sonuçlar
                                st.markdown("---")
                                st.subheader("📄 Sayfa Detayları")
                                
                                for result in all_results:
                                    with st.expander(f"Sayfa {result['page']} - Güven: {result['confidence']:.2%}"):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.image(
                                                result['image'],
                                                caption=f"Sayfa {result['page']}",
                                                use_container_width=True
                                            )
                                        
                                        with col2:
                                            st.text_area(
                                                "Tespit Edilen Metin:",
                                                value=result['text'],
                                                height=300,
                                                key=f"pdf_page_{result['page']}_text"
                                            )
                                            
                                            st.metric("Güven Skoru", f"{result['confidence']:.2%}")
                                
                            except Exception as e:
                                st.error(f"❌ PDF işleme hatası: {str(e)}")
                                import traceback
                                with st.expander("Hata Detayları"):
                                    st.code(traceback.format_exc())
                
                except Exception as e:
                    st.error(f"❌ PDF yükleme hatası: {str(e)}")
                    
                    # Backend bilgisi göster
                    backend_info = pdf_processor.get_backend_info()
                    st.info(
                        f"**PDF Backend:** {backend_info.get('backend', 'Bilinmiyor')}\n\n"
                        f"**Gerekli Kütüphane:** pdf2image veya PyMuPDF\n\n"
                        "**Kurulum:**\n"
                        "```bash\n"
                        "pip install pdf2image PyMuPDF\n"
                        "```"
                    )
        else:
            st.warning(f"Seçilen model başlatılmamış. Lütfen sidebar'dan modeli başlatın.")
    else:
        st.warning("Henüz model kaydedilmemiş!")

# TAB 3: Karşılaştırma
with tab3:
    st.header("Model Karşılaştırması")
    
    # Başlatılmış modelleri filtrele
    initialized_models = {
        model_id: registry.get_model(model_id)
        for model_id in model_ids
        if registry.get_model(model_id) and registry.get_model(model_id).is_initialized()
    }
    
    if not initialized_models:
        st.warning("Karşılaştırma için en az bir model başlatılmış olmalı!")
    else:
        # Model seçimi (çoklu)
        selected_models = st.multiselect(
            "Karşılaştırılacak Modelleri Seçin:",
            options=list(initialized_models.keys()),
            default=list(initialized_models.keys())[:2] if len(initialized_models) >= 2 else list(initialized_models.keys()),
            key="comparison_model_select"
        )
        
        if selected_models:
            # Görüntü yükleme
            comparison_image = st.file_uploader(
                "Karşılaştırma için görüntü yükleyin:",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key="comparison_image_upload"
            )
            
            if comparison_image is not None:
                image = Image.open(comparison_image)
                
                # Görüntüyü göster
                st.subheader("Test Görüntüsü")
                st.image(image, width='stretch')
                
                # Karşılaştırma butonu
                if st.button("🔄 Modelleri Karşılaştır", key="compare_button"):
                    # Seçili modelleri al
                    models_to_compare = {
                        model_id: initialized_models[model_id]
                        for model_id in selected_models
                    }
                    
                    with st.spinner("Modeller karşılaştırılıyor..."):
                        # Karşılaştırma yap
                        comparator = ModelComparator()
                        comparison_results = comparator.compare_models(
                            models_to_compare,
                            image,
                            include_timing=True
                        )
                        
                        st.session_state.comparison_results = comparison_results
                        
                        # Metrikleri hesapla
                        metrics = comparator.calculate_metrics(comparison_results)
                        
                        # Sonuçları göster
                        st.subheader("📊 Karşılaştırma Sonuçları")
                        
                        # Metrikler tablosu
                        import pandas as pd
                        
                        metrics_data = []
                        for model_id, metric in metrics.items():
                            result_data = comparison_results.get(model_id, {})
                            metrics_data.append({
                                "Model": model_id,
                                "Başarılı": "✅" if result_data.get("success") else "❌",
                                "Metin Uzunluğu": metric.get("text_length", 0),
                                "Kelime Sayısı": metric.get("word_count", 0),
                                "Ortalama Güven": f"{metric.get('avg_confidence', 0):.2%}",
                                "İşlem Süresi (s)": f"{metric.get('processing_time', 0):.3f}"
                            })
                        
                        df = pd.DataFrame(metrics_data)
                        st.dataframe(df, width='stretch')
                        
                        # Her model için sonuçlar
                        st.subheader("📝 Model Sonuçları")
                        
                        cols = st.columns(len(selected_models))
                        
                        for idx, model_id in enumerate(selected_models):
                            with cols[idx]:
                                st.markdown(f"### {model_id}")
                                
                                result_data = comparison_results.get(model_id, {})
                                
                                if result_data.get("success"):
                                    ocr_result = result_data.get("result")
                                    
                                    st.text_area(
                                        "Tespit Edilen Metin:",
                                        value=ocr_result.text if ocr_result else "",
                                        height=250,
                                        key=f"result_{model_id}"
                                    )
                                    
                                    st.metric(
                                        "Güven Skoru",
                                        f"{ocr_result.confidence:.2%}" if ocr_result else "N/A"
                                    )
                                    
                                    st.metric(
                                        "İşlem Süresi",
                                        f"{result_data.get('time', 0):.3f}s"
                                    )
                                    
                                    # Bounding box görselleştirme
                                    if ocr_result and ocr_result.bboxes:
                                        processor = ImageProcessor()
                                        visualized = processor.draw_bounding_boxes(
                                            image,
                                            ocr_result.bboxes,
                                            colors=[f"hsl({idx * 60}, 70%, 50%)"]
                                        )
                                        st.image(visualized, width='stretch')
                                else:
                                    st.error(f"Hata: {result_data.get('error', 'Bilinmeyen hata')}")
                        
                        # En iyi model
                        best_model = comparator.get_best_model(metrics, metric="avg_confidence")
                        if best_model:
                            st.success(f"🏆 En yüksek güven skoruna sahip model: **{best_model}**")

# TAB 4: Bilgi
with tab4:
    st.header("ℹ️ Sistem Bilgileri")
    
    st.subheader("Kayıtlı Modeller")
    registry = st.session_state.registry
    
    for model_id in model_ids:
        model = registry.get_model(model_id)
        if model:
            with st.expander(f"📦 {model_id}"):
                info = model.get_model_info()
                st.json(info)
    
    st.markdown("---")
    st.subheader("Kullanım Kılavuzu")
    st.markdown("""
    1. **Model Başlatma**: Sidebar'dan modelleri başlatın
    2. **Image OCR**: Bir model seçip görüntü yükleyerek test edin
    3. **PDF OCR**: PDF belgelerini sayfa sayfa OCR ile işleyin
    4. **Karşılaştırma**: Birden fazla modeli seçip karşılaştırın
    5. **Yeni Model Ekleme**: `config.py` dosyasına yeni model yapılandırması ekleyin
    """)
    
    st.markdown("---")
    st.subheader("Desteklenen Platformlar")
    st.markdown("""
    - **HuggingFace**: Transformers kütüphanesi ile
    - **Ollama**: Yerel Ollama sunucusu ile
    - **PaddleOCR**: PaddleOCR kütüphanesi ile
    """)
    
    st.markdown("---")
    st.subheader("PDF İşleme")
    pdf_processor = st.session_state.pdf_processor
    backend_info = pdf_processor.get_backend_info()
    st.info(f"""
    **Backend**: {backend_info.get('backend', 'Bilinmiyor')}  
    **DPI**: {backend_info.get('dpi', 300)}  
    **Format**: {backend_info.get('format', 'RGB')}
    
    PDF işleme için gerekli kütüphaneler:
    - `pdf2image` (önerilen) - Poppler gerektirir
    - `PyMuPDF` (alternatif) - Ekstra bağımlılık gerektirmez
    """)

