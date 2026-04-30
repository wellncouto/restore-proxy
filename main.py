import os
import io
import base64
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client, handle_file
from PIL import Image, ImageDraw, ImageFont, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("restore-proxy")

# Spaces
RESTORE_SPACE = os.getenv(
    "RESTORE_SPACE",
    "avans06/Image_Face_Upscale_Restoration-GFPGAN-RestoreFormer-CodeFormer-GPEN",
)
COLORIZE_SPACE = os.getenv("COLORIZE_SPACE", "gudada/DDColor")

DEFAULT_FACE_MODEL = os.getenv("FACE_MODEL", "GFPGANv1.4.pth")
DEFAULT_UPSCALE_MODEL = os.getenv("UPSCALE_MODEL", "SRVGG, realesr-general-x4v3.pth")
DEFAULT_SCALE = float(os.getenv("SCALE", "2"))
GRAYSCALE_THRESHOLD = float(os.getenv("GRAYSCALE_THRESHOLD", "8"))  # avg channel diff

# Gemini (Nano Banana)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-image")

app = FastAPI(title="Restore Proxy", version="2.1")

# Inclui módulo do produto Álbum Pra Colorir
try:
    from colorir import router as colorir_router
    app.include_router(colorir_router)
    log.info("colorir router registrado")
except Exception as e:
    log.warning(f"colorir router não carregado: {e}")


# ───────── Watermark ─────────

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def add_watermark(
    img_bytes: bytes,
    watermark_text: str = "AMOSTRA",
    footer_text: str = "Pague o PIX para receber a foto sem marca d'água em alta qualidade",
    brand_text: str = "",
    max_size: int = 720,
    jpeg_quality: int = 70,
) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    # Aplica rotação baseada em EXIF (corrige fotos retrato que viriam deitadas)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGBA")

    # Reduz pra preview (incentiva pagar pelo HD)
    if max_size and max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    w, h = img.size

    # Diagonal repeated watermark
    diag_w, diag_h = int(w * 1.5), int(h * 1.5)
    overlay = Image.new("RGBA", (diag_w, diag_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(36, w // 14)
    font = _load_font(font_size)

    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    spacing_x = max(text_w + w // 10, w // 4)
    spacing_y = max(text_h + h // 8, h // 4)

    for y in range(0, diag_h, spacing_y):
        offset = (y // spacing_y) % 2 * (spacing_x // 2)
        for x in range(-spacing_x, diag_w, spacing_x):
            draw.text((x + offset, y), watermark_text, fill=(255, 255, 255, 110), font=font)
            draw.text(
                (x + offset + 2, y + 2), watermark_text, fill=(0, 0, 0, 80), font=font
            )

    overlay = overlay.rotate(-30, resample=Image.BICUBIC, expand=False)
    crop_x = (diag_w - w) // 2
    crop_y = (diag_h - h) // 2
    overlay = overlay.crop((crop_x, crop_y, crop_x + w, crop_y + h))

    img_with_wm = Image.alpha_composite(img, overlay)

    # Footer banner
    banner_h = max(60, h // 10)
    banner = Image.new("RGBA", (w, banner_h), (15, 23, 42, 235))
    banner_draw = ImageDraw.Draw(banner)
    footer_font = _load_font(max(18, banner_h // 3))

    bbox = banner_draw.textbbox((0, 0), footer_text, font=footer_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # auto-shrink se não couber
    while tw > w * 0.92 and footer_font.size > 14:
        new_size = footer_font.size - 2
        footer_font = _load_font(new_size)
        bbox = banner_draw.textbbox((0, 0), footer_text, font=footer_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    banner_draw.text(((w - tw) // 2, (banner_h - th) // 2 - 2), footer_text,
                     fill=(255, 255, 255, 255), font=footer_font)

    # Top brand bar (opcional)
    top_h = 0
    if brand_text:
        top_h = max(50, h // 14)
        top_bar = Image.new("RGBA", (w, top_h), (15, 23, 42, 235))
        td = ImageDraw.Draw(top_bar)
        brand_font = _load_font(max(20, top_h // 2))
        bb = td.textbbox((0, 0), brand_text, font=brand_font)
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        td.text(((w - bw) // 2, (top_h - bh) // 2 - 2), brand_text,
                fill=(255, 255, 255, 255), font=brand_font)

    final_h = h + banner_h + top_h
    final = Image.new("RGB", (w, final_h), (15, 23, 42))
    if top_h:
        final.paste(top_bar.convert("RGB"), (0, 0))
    final.paste(img_with_wm.convert("RGB"), (0, top_h))
    final.paste(banner.convert("RGB"), (0, top_h + h))

    out = io.BytesIO()
    final.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
    return out.getvalue()


class WatermarkIn(BaseModel):
    image_b64: str
    watermark_text: Optional[str] = "AMOSTRA"
    footer_text: Optional[str] = (
        "Pague o PIX para receber a foto sem marca d'água em alta qualidade"
    )
    brand_text: Optional[str] = ""
    max_size: Optional[int] = 720
    jpeg_quality: Optional[int] = 70


class WatermarkOut(BaseModel):
    image_b64: str
    mime_type: str = "image/jpeg"


@app.post("/watermark", response_model=WatermarkOut)
def watermark(body: WatermarkIn):
    raw = body.image_b64
    if "," in raw and raw.lstrip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw)
    except Exception as e:
        raise HTTPException(400, f"image_b64 inválido: {e}")
    if len(img_bytes) < 1024:
        raise HTTPException(400, "imagem muito pequena")

    out_bytes = add_watermark(
        img_bytes,
        watermark_text=body.watermark_text or "AMOSTRA",
        footer_text=body.footer_text or "",
        brand_text=body.brand_text or "",
        max_size=body.max_size or 720,
        jpeg_quality=body.jpeg_quality or 70,
    )
    return WatermarkOut(image_b64=base64.b64encode(out_bytes).decode("ascii"))

_restore_client: Optional[Client] = None
_colorize_client: Optional[Client] = None


def _build_client(slug: str) -> Client:
    """Cria Client com timeouts longos pra Spaces em CPU."""
    try:
        return Client(slug, httpx_kwargs={"timeout": 600})
    except TypeError:
        # versão antiga do gradio_client sem httpx_kwargs
        c = Client(slug)
        try:
            import httpx
            c.httpx_kwargs = {"timeout": 600}
        except Exception:
            pass
        return c


def get_restore_client() -> Client:
    global _restore_client
    if _restore_client is None:
        log.info("connecting RESTORE space=%s", RESTORE_SPACE)
        _restore_client = _build_client(RESTORE_SPACE)
    return _restore_client


def get_colorize_client() -> Client:
    global _colorize_client
    if _colorize_client is None:
        log.info("connecting COLORIZE space=%s", COLORIZE_SPACE)
        _colorize_client = _build_client(COLORIZE_SPACE)
    return _colorize_client


def is_grayscale(img_bytes: bytes) -> bool:
    """Detecta se a foto é P&B (canais R/G/B muito próximos)."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        small = img.resize((64, 64))
        pixels = list(small.getdata())
        diffs = [max(p) - min(p) for p in pixels]
        avg = sum(diffs) / len(diffs)
        log.info("grayscale check: avg_channel_diff=%.2f (threshold=%.1f)", avg, GRAYSCALE_THRESHOLD)
        return avg < GRAYSCALE_THRESHOLD
    except Exception as e:
        log.warning("grayscale check fail: %s", e)
        return False


def colorize_step(input_path: str) -> str:
    """Chama DDColor. Retorna path da imagem colorizada."""
    client = get_colorize_client()
    log.info("colorize via DDColor...")
    t0 = time.time()
    result = client.predict(img=handle_file(input_path), api_name="/colorize")
    log.info("colorize done in %.1fs", time.time() - t0)
    # result = (before_path, after_path)
    if not result or not isinstance(result, (list, tuple)) or len(result) < 2 or not result[1]:
        raise HTTPException(502, f"DDColor retornou inesperado: {result}")
    return result[1]


def restore_step(input_path: str, face_model: str, upscale_model: str, scale: float) -> str:
    """Chama GFPGAN+RealESR. Retorna path da imagem final restaurada."""
    client = get_restore_client()
    gallery = [{"image": handle_file(input_path), "caption": None}]
    log.info("restore: face=%s upscale=%s scale=%s", face_model, upscale_model, scale)
    t0 = time.time()
    result = client.predict(
        gallery,
        face_model,
        upscale_model,
        scale,
        "retinaface_resnet50",
        10,
        False,
        True,
        True,
        api_name="/inference",
    )
    log.info("restore done in %.1fs", time.time() - t0)
    out_gallery, _ = result
    if not out_gallery:
        raise HTTPException(502, "Space restore retornou galeria vazia")

    final_item = out_gallery[-1]  # último = final composto
    img_field = final_item.get("image") if isinstance(final_item, dict) else final_item
    if isinstance(img_field, dict):
        path = img_field.get("path")
    elif isinstance(img_field, str):
        path = img_field
    else:
        raise HTTPException(502, f"output inesperado: {out_gallery}")
    if not path or not os.path.exists(path):
        raise HTTPException(502, f"arquivo final não existe: {path}")
    return path


class RestoreIn(BaseModel):
    image_b64: str
    face_model: Optional[str] = None
    upscale_model: Optional[str] = None
    scale: Optional[float] = None
    force_colorize: Optional[bool] = None      # força colorizar mesmo se colorida
    skip_colorize: Optional[bool] = None       # pula colorização


class RestoreOut(BaseModel):
    image_b64: str
    mime_type: str
    face_model: str
    upscale_model: str
    scale: float
    duration_s: float
    colorized: bool
    detected_grayscale: bool


@app.get("/health")
def health():
    return {
        "ok": True,
        "restore_space": RESTORE_SPACE,
        "colorize_space": COLORIZE_SPACE,
    }


@app.post("/restore", response_model=RestoreOut)
def restore(body: RestoreIn):
    face_model = body.face_model or DEFAULT_FACE_MODEL
    upscale_model = body.upscale_model or DEFAULT_UPSCALE_MODEL
    scale = body.scale or DEFAULT_SCALE

    raw = body.image_b64
    if "," in raw and raw.lstrip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw)
    except Exception as e:
        raise HTTPException(400, f"image_b64 inválido: {e}")
    if len(img_bytes) < 4096:
        raise HTTPException(400, "imagem muito pequena (<4KB) — Space rejeita")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(img_bytes)
        in_path = f.name

    t_total = time.time()
    try:
        # 1) Decisão de colorização
        is_bw = is_grayscale(img_bytes)
        will_colorize = (
            body.force_colorize is True
            or (is_bw and body.skip_colorize is not True)
        )

        current_path = in_path
        colorized = False

        if will_colorize:
            try:
                current_path = colorize_step(in_path)
                colorized = True
            except Exception as e:
                log.warning("colorize falhou, segue sem colorizar: %s", e)
                global _colorize_client
                _colorize_client = None

        # 2) Restauração + upscale (sempre)
        final_path = restore_step(current_path, face_model, upscale_model, scale)

        out_bytes = Path(final_path).read_bytes()
        out_b64 = base64.b64encode(out_bytes).decode("ascii")
        ext = Path(final_path).suffix.lstrip(".").lower() or "png"
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")

        return RestoreOut(
            image_b64=out_b64,
            mime_type=mime,
            face_model=face_model,
            upscale_model=upscale_model,
            scale=scale,
            duration_s=round(time.time() - t_total, 2),
            colorized=colorized,
            detected_grayscale=is_bw,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("falha no restore")
        # reset clients pra forçar reconexão
        global _restore_client
        _restore_client = None
        raise HTTPException(502, f"falha no restore: {e}")
    finally:
        try:
            os.unlink(in_path)
        except Exception:
            pass


# ───────── Gemini (Nano Banana) ─────────

DEFAULT_GEMINI_PROMPT = (
    "Restore and naturally colorize this old damaged photograph. "
    "Repair scratches, tears, stains, dust, fading and discoloration. "
    "If black and white or sepia, colorize with realistic period-accurate skin tones, "
    "eye colors, hair, clothing and background colors. "
    "Sharpen all details and improve overall quality dramatically. "
    "Keep the EXACT SAME people, faces, identities, expressions, clothing patterns, "
    "body postures, background composition and pose — do NOT change who they are or "
    "add/remove anything. Output a high quality professionally restored version of "
    "the original photo."
)


class GeminiIn(BaseModel):
    image_b64: str
    prompt: Optional[str] = None
    model: Optional[str] = None


class GeminiOut(BaseModel):
    image_b64: str
    mime_type: str
    duration_s: float
    model: str


@app.post("/restore-gemini", response_model=GeminiOut)
def restore_gemini(body: GeminiIn):
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY não configurada no ambiente")

    raw = body.image_b64
    if "," in raw and raw.lstrip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw)
    except Exception as e:
        raise HTTPException(400, f"image_b64 inválido: {e}")
    if len(img_bytes) < 1024:
        raise HTTPException(400, "imagem muito pequena")

    # Detecta mime
    try:
        img = Image.open(io.BytesIO(img_bytes))
        fmt = (img.format or "JPEG").upper()
        in_mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(fmt, "image/jpeg")
    except Exception:
        in_mime = "image/jpeg"

    model = body.model or GEMINI_MODEL
    prompt = body.prompt or DEFAULT_GEMINI_PROMPT
    in_b64 = base64.b64encode(img_bytes).decode("ascii")

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": in_mime, "data": in_b64}}
            ]
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"]
        }
    }

    import httpx
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    log.info("gemini call model=%s in_size=%dKB", model, len(img_bytes) // 1024)
    t0 = time.time()
    try:
        with httpx.Client(timeout=180) as h:
            r = h.post(
                url,
                headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        body_text = e.response.text[:600] if e.response is not None else ""
        raise HTTPException(502, f"gemini HTTP {e.response.status_code}: {body_text}")
    except Exception as e:
        raise HTTPException(502, f"gemini falha: {e}")

    duration = time.time() - t0

    # Extrai imagem do retorno
    candidates = data.get("candidates", [])
    if not candidates:
        raise HTTPException(502, f"gemini sem candidates: {str(data)[:400]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    out_b64 = None
    out_mime = "image/png"
    for p in parts:
        inline = p.get("inlineData") or p.get("inline_data")
        if inline and inline.get("data"):
            out_b64 = inline["data"]
            out_mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            break
    if not out_b64:
        raise HTTPException(502, f"gemini não retornou imagem: {str(data)[:400]}")

    return GeminiOut(
        image_b64=out_b64,
        mime_type=out_mime,
        duration_s=round(duration, 2),
        model=model,
    )


# ───────── Rotate Fix (EXIF) ─────────

class RotateFixIn(BaseModel):
    image_b64: str
    output_format: Optional[str] = "jpeg"   # jpeg | png
    output_quality: Optional[int] = 95


class RotateFixOut(BaseModel):
    image_b64: str
    mime_type: str
    width: int
    height: int


@app.post("/rotate-fix", response_model=RotateFixOut)
def rotate_fix(body: RotateFixIn):
    raw = body.image_b64
    if "," in raw and raw.lstrip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw)
    except Exception as e:
        raise HTTPException(400, f"image_b64 inválido: {e}")
    if len(img_bytes) < 1024:
        raise HTTPException(400, "imagem muito pequena")

    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img)

    fmt = (body.output_format or "jpeg").lower()
    if fmt == "png":
        img = img.convert("RGBA")
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        mime = "image/png"
    else:
        img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=body.output_quality or 95, optimize=True)
        mime = "image/jpeg"

    return RotateFixOut(
        image_b64=base64.b64encode(out.getvalue()).decode("ascii"),
        mime_type=mime,
        width=img.width,
        height=img.height,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
