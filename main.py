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
from PIL import Image

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

app = FastAPI(title="Restore Proxy", version="2.0")

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
