import os
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client, handle_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("restore-proxy")

SPACE = os.getenv(
    "HF_SPACE",
    "avans06/Image_Face_Upscale_Restoration-GFPGAN-RestoreFormer-CodeFormer-GPEN",
)
HF_TOKEN = os.getenv("HF_TOKEN") or None

DEFAULT_FACE_MODEL = os.getenv("FACE_MODEL", "GFPGANv1.4.pth")
DEFAULT_UPSCALE_MODEL = os.getenv("UPSCALE_MODEL", "SRVGG, realesr-general-x4v3.pth")
DEFAULT_SCALE = float(os.getenv("SCALE", "2"))
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "240"))

app = FastAPI(title="Restore Proxy", version="1.1")

_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        log.info("connecting to space=%s", SPACE)
        kwargs = {}
        if HF_TOKEN:
            # gradio_client < 2.0 uses hf_token; >= 2.0 uses headers/auth
            try:
                kwargs["hf_token"] = HF_TOKEN
                _client = Client(SPACE, **kwargs)
            except TypeError:
                _client = Client(SPACE, headers={"Authorization": f"Bearer {HF_TOKEN}"})
        else:
            _client = Client(SPACE)
        log.info("connected.")
    return _client


class RestoreIn(BaseModel):
    image_b64: str
    face_model: Optional[str] = None
    upscale_model: Optional[str] = None
    scale: Optional[float] = None


class RestoreOut(BaseModel):
    image_b64: str
    mime_type: str
    face_model: str
    upscale_model: str
    scale: float
    duration_s: float


@app.get("/health")
def health():
    return {"ok": True, "space": SPACE}


@app.post("/restore", response_model=RestoreOut)
def restore(body: RestoreIn):
    import time

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

    try:
        client = get_client()

        gallery_input = [{"image": handle_file(in_path), "caption": None}]

        log.info(
            "inference face=%s upscale=%s scale=%s size=%dKB",
            face_model, upscale_model, scale, len(img_bytes) // 1024,
        )
        t0 = time.time()
        result = client.predict(
            gallery_input,
            face_model,
            upscale_model,
            scale,
            "retinaface_resnet50",  # face_detection
            10,                      # face_detection_threshold
            False,                   # face_detection_only_center
            True,                    # outputwithmodelname
            True,                    # save_as_png
            api_name="/inference",
        )
        duration = time.time() - t0
        log.info("inference done in %.1fs", duration)

        out_gallery, _ = result
        if not out_gallery:
            raise HTTPException(502, "Space retornou galeria vazia")

        # The Space returns 4 outputs in order:
        #   [0] cropped_faces, [1] restored_faces, [2] cmp (before/after grid),
        #   [3] FINAL composed image (face restored + bg upscaled). We want [3].
        final_item = out_gallery[-1]  # last = full final image

        # gradio gallery item can be either:
        #   {"image": {"path":..., "url":...}, "caption": None}  (newer)
        #   {"image": "/some/path", "caption": None}             (older / downloaded)
        #   "/some/path"                                          (raw string)
        img_field = final_item.get("image") if isinstance(final_item, dict) else final_item
        if isinstance(img_field, dict):
            path = img_field.get("path")
            url = img_field.get("url")
            name = img_field.get("orig_name") or path or ""
        elif isinstance(img_field, str):
            path = img_field
            url = None
            name = os.path.basename(img_field)
        else:
            raise HTTPException(502, f"output inesperado: {out_gallery}")
        out_bytes = None
        if path and os.path.exists(path):
            out_bytes = Path(path).read_bytes()
        elif url:
            import httpx
            with httpx.Client(timeout=60) as h:
                r = h.get(url)
                r.raise_for_status()
                out_bytes = r.content

        if not out_bytes:
            raise HTTPException(502, "não consegui ler a imagem final")

        out_b64 = base64.b64encode(out_bytes).decode("ascii")
        ext = (Path(name).suffix or ".png").lstrip(".").lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")

        return RestoreOut(
            image_b64=out_b64,
            mime_type=mime,
            face_model=face_model,
            upscale_model=upscale_model,
            scale=scale,
            duration_s=round(duration, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("falha no restore")
        global _client
        _client = None  # força reconexão na próxima chamada
        raise HTTPException(502, f"falha no restore: {e}")
    finally:
        try:
            os.unlink(in_path)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
