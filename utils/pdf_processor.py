"""
PDF İşleme Modülü

Bu modül PDF dosyalarını sayfa sayfa görüntülere dönüştürür ve OCR için hazırlar.
"""

from typing import List, Optional, Tuple
from PIL import Image
import io
import tempfile
import os


class PDFProcessor:
    """PDF dosyalarını işlemek için utility sınıfı"""
    
    def __init__(self, dpi: int = 300, format: str = "RGB"):
        """
        Args:
            dpi: PDF'den görüntü oluştururken kullanılacak DPI (yüksek=kaliteli ama yavaş)
            format: Görüntü formatı ('RGB', 'L' (grayscale), vb.)
        """
        self.dpi = dpi
        self.format = format
        self._backend = self._detect_backend()
    
    def _detect_backend(self) -> str:
        """Kullanılabilir PDF backend'ini tespit et"""
        # Önce pdf2image dene (poppler tabanlı, en iyi sonuç)
        try:
            import pdf2image
            return "pdf2image"
        except ImportError:
            pass
        
        # Fallback: PyMuPDF (fitz)
        try:
            import fitz
            return "pymupdf"
        except ImportError:
            pass
        
        raise ImportError(
            "PDF işleme için gerekli kütüphane bulunamadı!\n"
            "Lütfen şunlardan birini kurun:\n"
            "  pip install pdf2image  (önerilen)\n"
            "  pip install PyMuPDF    (alternatif)\n\n"
            "Not: pdf2image için poppler da gereklidir:\n"
            "  Windows: https://github.com/oschwartz10612/poppler-windows/releases/\n"
            "  Linux: sudo apt-get install poppler-utils\n"
            "  macOS: brew install poppler"
        )
    
    def pdf_to_images(
        self, 
        pdf_file, 
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
        max_pages: Optional[int] = None
    ) -> List[Image.Image]:
        """
        PDF dosyasını sayfa sayfa görüntülere dönüştürür
        
        Args:
            pdf_file: PDF dosyası (bytes, file-like object, veya dosya yolu)
            first_page: İşlenecek ilk sayfa (1'den başlar, None=baştan başla)
            last_page: İşlenecek son sayfa (None=sona kadar)
            max_pages: Maksimum işlenecek sayfa sayısı (None=sınırsız)
        
        Returns:
            PIL Image nesnelerinin listesi (her biri bir PDF sayfası)
        """
        if self._backend == "pdf2image":
            return self._pdf_to_images_pdf2image(pdf_file, first_page, last_page, max_pages)
        elif self._backend == "pymupdf":
            return self._pdf_to_images_pymupdf(pdf_file, first_page, last_page, max_pages)
        else:
            raise RuntimeError("PDF backend bulunamadı")
    
    def _pdf_to_images_pdf2image(
        self, 
        pdf_file, 
        first_page: Optional[int],
        last_page: Optional[int],
        max_pages: Optional[int]
    ) -> List[Image.Image]:
        """pdf2image backend ile PDF'i görüntülere dönüştür"""
        from pdf2image import convert_from_bytes, convert_from_path
        
        try:
            # Dosya yolu mu bytes mı kontrol et
            if isinstance(pdf_file, (str, os.PathLike)):
                # Dosya yolu
                images = convert_from_path(
                    pdf_file,
                    dpi=self.dpi,
                    first_page=first_page,
                    last_page=last_page,
                    fmt=self.format.lower()
                )
            else:
                # Bytes veya file-like object
                if hasattr(pdf_file, 'read'):
                    pdf_bytes = pdf_file.read()
                else:
                    pdf_bytes = pdf_file
                
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=self.dpi,
                    first_page=first_page,
                    last_page=last_page,
                    fmt=self.format.lower()
                )
            
            # max_pages sınırlaması uygula
            if max_pages is not None:
                images = images[:max_pages]
            
            return images
            
        except Exception as e:
            raise Exception(f"PDF'i görüntülere dönüştürme hatası (pdf2image): {str(e)}")
    
    def _pdf_to_images_pymupdf(
        self, 
        pdf_file, 
        first_page: Optional[int],
        last_page: Optional[int],
        max_pages: Optional[int]
    ) -> List[Image.Image]:
        """PyMuPDF (fitz) backend ile PDF'i görüntülere dönüştür"""
        import fitz
        
        try:
            # Dosya yolu mu bytes mı kontrol et
            if isinstance(pdf_file, (str, os.PathLike)):
                doc = fitz.open(pdf_file)
            else:
                # Bytes veya file-like object
                if hasattr(pdf_file, 'read'):
                    pdf_bytes = pdf_file.read()
                else:
                    pdf_bytes = pdf_file
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Sayfa aralığını belirle
            total_pages = len(doc)
            start_page = (first_page - 1) if first_page else 0
            end_page = (last_page) if last_page else total_pages
            
            # max_pages sınırlaması
            if max_pages is not None:
                end_page = min(end_page, start_page + max_pages)
            
            images = []
            
            # DPI'yi zoom faktörüne çevir (72 DPI = 1.0 zoom)
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Pixmap'i PIL Image'e dönüştür
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Format dönüşümü (eğer gerekirse)
                if img.mode != self.format:
                    img = img.convert(self.format)
                
                images.append(img)
            
            doc.close()
            return images
            
        except Exception as e:
            raise Exception(f"PDF'i görüntülere dönüştürme hatası (PyMuPDF): {str(e)}")
    
    def get_page_count(self, pdf_file) -> int:
        """PDF dosyasındaki toplam sayfa sayısını döndürür"""
        if self._backend == "pdf2image":
            # pdf2image sayfa sayısını doğrudan döndürmez, PyMuPDF kullan
            try:
                import fitz
                if isinstance(pdf_file, (str, os.PathLike)):
                    doc = fitz.open(pdf_file)
                else:
                    if hasattr(pdf_file, 'read'):
                        pdf_bytes = pdf_file.read()
                        # Dosya pointer'ını başa al
                        if hasattr(pdf_file, 'seek'):
                            pdf_file.seek(0)
                    else:
                        pdf_bytes = pdf_file
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_count = len(doc)
                doc.close()
                return page_count
            except:
                # PyMuPDF yoksa, tüm sayfaları dönüştürüp say
                images = self._pdf_to_images_pdf2image(pdf_file, None, None, None)
                return len(images)
        
        elif self._backend == "pymupdf":
            import fitz
            if isinstance(pdf_file, (str, os.PathLike)):
                doc = fitz.open(pdf_file)
            else:
                if hasattr(pdf_file, 'read'):
                    pdf_bytes = pdf_file.read()
                    # Dosya pointer'ını başa al
                    if hasattr(pdf_file, 'seek'):
                        pdf_file.seek(0)
                else:
                    pdf_bytes = pdf_file
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(doc)
            doc.close()
            return page_count
    
    def get_backend_info(self) -> dict:
        """Kullanılan backend hakkında bilgi döndürür"""
        info = {
            "backend": self._backend,
            "dpi": self.dpi,
            "format": self.format
        }
        
        if self._backend == "pdf2image":
            try:
                import pdf2image
                info["version"] = pdf2image.__version__
                info["requires_poppler"] = True
            except:
                pass
        elif self._backend == "pymupdf":
            try:
                import fitz
                info["version"] = fitz.__version__
                info["requires_poppler"] = False
            except:
                pass
        
        return info


# Convenience function
def convert_pdf_to_images(
    pdf_file,
    dpi: int = 300,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    max_pages: Optional[int] = None
) -> List[Image.Image]:
    """
    PDF dosyasını görüntülere dönüştüren yardımcı fonksiyon
    
    Args:
        pdf_file: PDF dosyası (bytes, file-like object, veya dosya yolu)
        dpi: Görüntü kalitesi (varsayılan: 300)
        first_page: İlk sayfa (1'den başlar)
        last_page: Son sayfa
        max_pages: Maksimum sayfa sayısı
    
    Returns:
        PIL Image listesi
    """
    processor = PDFProcessor(dpi=dpi)
    return processor.pdf_to_images(pdf_file, first_page, last_page, max_pages)

