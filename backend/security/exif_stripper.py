"""
Sanitize uploaded clinical dermoscopy images by stripping EXIF and patient metadata.
"""
import io
from PIL import Image

class EXIFMetadataStripper:
    def strip_metadata(self, image_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(image_bytes))
        clean_img = Image.new(img.mode, img.size)
        clean_img.putdata(list(img.getdata()))
        out_buf = io.BytesIO()
        clean_img.save(out_buf, format=img.format or "PNG")
        return out_buf.getvalue()
