"""
Endpoints do produto Álbum Pra Colorir.

Fluxo:
  1. POST /colorir/album/criar       → cria álbum, retorna token + url
  2. POST /colorir/album/{token}/upload → cliente envia 1 foto por vez
  3. POST /colorir/album/{token}/processar → clicou Enviar, dispara IA + monta PDF preview
  4. GET  /colorir/album/{token}/status → polling do frontend
  5. GET  /colorir/album/{token}/preview → serve PDF com watermark
  6. POST /colorir/album/{token}/liberar → n8n chama após PIX, gera PDF limpo
  7. GET  /colorir/historico/{phone}  → lista PDFs pagos pra "Meus Livros"
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import psycopg2
import psycopg2.extras
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel

log = logging.getLogger("colorir")

router = APIRouter(prefix="/colorir", tags=["colorir"])

# ───────── Config ─────────
DB_DSN = os.getenv("COLORIR_DB_DSN", "")
STORAGE_ROOT = Path(os.getenv("COLORIR_STORAGE_PATH", "/data/colorir"))
TOKEN_SECRET = os.getenv("COLORIR_TOKEN_SECRET", "change-me-in-prod")
OPENAI_KEY = os.getenv("COLORIR_OPENAI_KEY", os.getenv("OPENAI_API_KEY", ""))
BASE_URL = os.getenv("COLORIR_BASE_URL", "https://colorir.example.com")
PIX_CHAVE = os.getenv("COLORIR_PIX_CHAVE", "e58e968f-2c38-43a4-b094-5ecf0eefd21a")
WHATSAPP_NUM = os.getenv("COLORIR_WHATSAPP_NUM", "5547991100824")

OPENAI_EDITS_URL = "https://api.openai.com/v1/images/edits"
OPENAI_MODEL = "gpt-image-2"

# Limites
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_FOTOS_PACK = 25                  # pack premium
MIN_FOTOS_PACK = 1                   # capa só
ALLOWED_PACKS = (5, 10, 25)
PROCESS_PARALLEL = 4                 # quantas chamadas IA simultâneas

# Prompts
PROMPT_COLORIR = (
    "Convert this photo into a CHILDREN'S COLORING BOOK PAGE in PORTRAIT (vertical A4) orientation. "
    "STRICT REQUIREMENTS: only pure black continuous vector-style outlines on PURE WHITE background. "
    "Use UNIFORM thick (4-6 pixel) clean black stroke lines. NO cross-hatching, NO shading, NO grayscale, "
    "NO gradients, NO textures, NO dots, NO speckles, NO sketchy strokes. Each line must be solid, "
    "continuous and closed where possible — like a printed coloring book. "
    "Cartoon style with cute exaggerated friendly features, big eyes, simplified shapes (Disney/kawaii feel). "
    "BACKGROUND RULE: PRESERVE the actual background from the original photo (trees, beach, room, park, "
    "house, etc). DO NOT invent generic backgrounds (no flowers/butterflies/balloons unless they exist in "
    "the original). Stylize the real background as clean cartoon line art — simplify shapes, exaggerate "
    "characteristics, but KEEP the scene recognizable (if the photo is at a beach, draw waves and sand; "
    "if at a park, draw the trees/grass/playground; if indoor, draw the room elements like sofa, window). "
    "Goal: each page tells the story of where the family was. Compose to fill the entire A4 portrait nicely. "
    "Leave generous white space inside shapes for crayons. No text, no signature, no watermark."
)
PROMPT_PIXAR_CAPA = (
    "Convert this photo into a vibrant 3D Pixar/Disney style colorful cartoon illustration suitable for a "
    "children's book cover. Friendly cute exaggerated features, big expressive eyes, bright vibrant saturated "
    "colors, smooth shading, warm cheerful lighting. "
    "BACKGROUND RULE: PRESERVE the actual setting from the original photo — if it's outdoors (beach, park, "
    "mountains), keep nature elements; if indoor (living room, kitchen, bedroom), keep furniture/walls. "
    "DO NOT invent generic backgrounds. Stylize the real background as Pixar 3D — soft volumetric lighting, "
    "stylized props, depth of field with bokeh. The setting should feel real but enchanted. "
    "The subjects should look adorable and approachable, KEEPING facial features recognizable. "
    "Full color, NOT line art."
)

A4_PX = (2480, 3508)  # 300 DPI


# ───────── Helpers ─────────


def _db():
    if not DB_DSN:
        raise HTTPException(500, "COLORIR_DB_DSN não configurado")
    return psycopg2.connect(DB_DSN, cursor_factory=psycopg2.extras.RealDictCursor)


@contextmanager
def db_conn():
    conn = _db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _make_token(phone: str) -> str:
    """Token único: HMAC + nonce aleatório. 32 chars hex."""
    nonce = secrets.token_hex(8)
    raw = f"{phone}:{nonce}:{int(time.time())}"
    sig = hmac.new(TOKEN_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return sig[:32]


def _album_dir(token: str) -> Path:
    p = STORAGE_ROOT / token
    p.mkdir(parents=True, exist_ok=True)
    (p / "originals").mkdir(exist_ok=True)
    (p / "processed").mkdir(exist_ok=True)
    (p / "pdfs").mkdir(exist_ok=True)
    return p


def _record_event(album_id: int | None, phone: str | None, acao: str, payload: dict | None = None):
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO colorir.eventos (album_id, phone, acao, payload) VALUES (%s,%s,%s,%s)",
                    (album_id, phone, acao, json.dumps(payload or {})),
                )
    except Exception as e:
        log.warning(f"event log falhou: {e}")


def _normalize_for_openai(raw_bytes: bytes) -> bytes:
    """Converte qualquer formato pra PNG RGB <=1024px (gpt-image-2 aceita)."""
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()


def _vectorize_lineart(png_bytes: bytes, target_w: int = 2048) -> bytes:
    """Limpa line art via potrace: bitmap ruidoso → SVG vetorial → PNG re-renderizado.
    Resulta em linhas pretas perfeitamente lisas, sem textura ou ruído.
    Requer 'potrace' e 'rsvg-convert' instalados no container.
    """
    import subprocess
    import tempfile

    img = Image.open(io.BytesIO(png_bytes)).convert("L")  # grayscale

    # Threshold binário (135) — força preto e branco puro, mata cinzas/textura
    bw = img.point(lambda p: 0 if p < 135 else 255, mode="1")

    with tempfile.TemporaryDirectory() as td:
        bmp_path = Path(td) / "input.bmp"
        svg_path = Path(td) / "out.svg"
        png_path = Path(td) / "out.png"

        # Pillow salva 1-bit como BMP (potrace aceita)
        bw.save(bmp_path, "BMP")

        # potrace: bitmap → SVG vetorial. Parâmetros pra line art limpo:
        # -t 5  : suaviza ruído (despeckle áreas <5px)
        # -O 0.5: optimização do path (curvas mais suaves)
        # -s    : output SVG
        # -a 1.0: corner threshold suave
        try:
            subprocess.run(
                ["potrace", str(bmp_path), "-s", "-t", "5", "-O", "0.5",
                 "-a", "1.0", "-o", str(svg_path)],
                check=True, capture_output=True, timeout=30,
            )
        except FileNotFoundError:
            log.warning("potrace não instalado, retornando bitmap original")
            return png_bytes
        except subprocess.CalledProcessError as e:
            log.warning(f"potrace falhou: {e.stderr.decode()[:200]}, retornando original")
            return png_bytes

        # Renderiza SVG de volta pra PNG em alta resolução via rsvg-convert
        try:
            subprocess.run(
                ["rsvg-convert", "-w", str(target_w), "-b", "white",
                 "-o", str(png_path), str(svg_path)],
                check=True, capture_output=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            log.warning(f"rsvg-convert indisponível, retornando original: {e}")
            return png_bytes

        return png_path.read_bytes()


def _call_openai_edit(image_bytes: bytes, prompt: str, size: str = "1024x1536", quality: str = "low") -> bytes:
    """Chama gpt-image-2 edits. Retorna PNG bytes."""
    if not OPENAI_KEY:
        raise HTTPException(500, "COLORIR_OPENAI_KEY não configurado")
    files = {"image": ("input.png", image_bytes, "image/png")}
    data = {"model": OPENAI_MODEL, "prompt": prompt, "size": size, "quality": quality}
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    with httpx.Client(timeout=180.0) as client:
        r = client.post(OPENAI_EDITS_URL, headers=headers, data=data, files=files)
    if r.status_code != 200:
        raise HTTPException(502, f"openai erro: {r.status_code} {r.text[:300]}")
    body = r.json()
    if "data" not in body or not body["data"]:
        raise HTTPException(502, f"openai sem data: {body}")
    return base64.b64decode(body["data"][0]["b64_json"])


# ───────── Schemas ─────────


class CriarAlbumIn(BaseModel):
    phone: str
    qtd_fotos: int = 10
    nome: Optional[str] = None
    capa_estilo: Optional[str] = "familia"


class CriarAlbumOut(BaseModel):
    token: str
    url: str
    status: str
    qtd_fotos: int
    valor_centavos: int


class StatusOut(BaseModel):
    token: str
    status: str
    qtd_fotos: int
    fotos_uploaded: int
    nome: Optional[str]
    capa_estilo: Optional[str]
    valor_centavos: int
    pdf_preview_url: Optional[str]
    pdf_final_url: Optional[str]
    fotos: list
    pix_chave: str
    whatsapp_url: str


class ProcessarAlbumIn(BaseModel):
    nome: Optional[str] = None
    capa_estilo: Optional[str] = None


class LiberarIn(BaseModel):
    valor_pago: Optional[int] = None
    msg_id_comprovante: Optional[str] = None


# ───────── Endpoints ─────────


@router.get("/_debug")
def debug():
    """Diagnóstico: testa env + conexão DB sem expor credenciais."""
    out = {
        "dsn_set": bool(DB_DSN),
        "dsn_format_ok": False,
        "dsn_host": None,
        "dsn_db": None,
        "storage_root_exists": STORAGE_ROOT.exists(),
        "storage_writable": False,
        "openai_key_set": bool(OPENAI_KEY),
        "openai_key_prefix": OPENAI_KEY[:8] if OPENAI_KEY else None,
        "token_secret_set": bool(TOKEN_SECRET) and TOKEN_SECRET != "change-me-in-prod",
        "base_url": BASE_URL,
        "db_connect_ok": False,
        "db_error": None,
        "schema_colorir_exists": False,
    }
    try:
        # parse DSN sem expor senha
        from urllib.parse import urlparse
        u = urlparse(DB_DSN)
        out["dsn_format_ok"] = u.scheme in ("postgres", "postgresql") and bool(u.hostname) and bool(u.path)
        out["dsn_host"] = u.hostname
        out["dsn_db"] = u.path.lstrip("/") if u.path else None
    except Exception as e:
        out["db_error"] = f"dsn parse: {e}"
        return out

    try:
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        (STORAGE_ROOT / ".write_test").write_text("ok")
        out["storage_writable"] = True
        (STORAGE_ROOT / ".write_test").unlink(missing_ok=True)
    except Exception as e:
        out["db_error"] = f"storage: {e}"

    try:
        conn = psycopg2.connect(DB_DSN, connect_timeout=5)
        out["db_connect_ok"] = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name='colorir'")
            out["schema_colorir_exists"] = cur.fetchone() is not None
        conn.close()
    except Exception as e:
        # mascarar qualquer ocorrência de senha (entre : e @ na DSN)
        msg = str(e)
        try:
            from urllib.parse import urlparse
            u = urlparse(DB_DSN)
            if u.password:
                msg = msg.replace(u.password, "***")
        except Exception:
            pass
        out["db_error"] = msg[:500]
    return out


@router.post("/album/criar", response_model=CriarAlbumOut)
def criar_album(body: CriarAlbumIn):
    if body.qtd_fotos not in ALLOWED_PACKS:
        raise HTTPException(400, f"qtd_fotos inválido. Permitidos: {ALLOWED_PACKS}")
    valor_map = {5: 1990, 10: 2900, 25: 5900}
    valor = valor_map[body.qtd_fotos]

    with db_conn() as conn:
        with conn.cursor() as cur:
            # 1 álbum ativo por phone
            cur.execute(
                """
                SELECT id, token, status, qtd_fotos, valor_centavos
                FROM colorir.albuns
                WHERE phone = %s AND status NOT IN ('PAGO','EXPIRADO','CANCELADO')
                ORDER BY criado_em DESC LIMIT 1
                """,
                (body.phone,),
            )
            existing = cur.fetchone()
            if existing:
                return CriarAlbumOut(
                    token=existing["token"],
                    url=f"{BASE_URL}/{existing['token']}",
                    status=existing["status"],
                    qtd_fotos=existing["qtd_fotos"],
                    valor_centavos=existing["valor_centavos"],
                )

            token = _make_token(body.phone)
            cur.execute(
                """
                INSERT INTO colorir.albuns
                  (token, phone, nome, capa_estilo, qtd_fotos, valor_centavos, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'NOVO')
                RETURNING id, token, status, qtd_fotos, valor_centavos
                """,
                (token, body.phone, body.nome, body.capa_estilo, body.qtd_fotos, valor),
            )
            row = cur.fetchone()
            _album_dir(token)

    _record_event(row["id"], body.phone, "LINK_CRIADO", {"qtd_fotos": body.qtd_fotos})
    return CriarAlbumOut(
        token=row["token"],
        url=f"{BASE_URL}/{row['token']}",
        status=row["status"],
        qtd_fotos=row["qtd_fotos"],
        valor_centavos=row["valor_centavos"],
    )


@router.post("/album/{token}/upload")
async def upload_foto(
    token: str,
    posicao: int = Form(...),
    eh_capa: bool = Form(False),
    file: UploadFile = File(...),
):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"foto muito grande (máx {MAX_UPLOAD_BYTES // (1024*1024)}MB)")

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, status, qtd_fotos FROM colorir.albuns WHERE token = %s", (token,))
            album = cur.fetchone()
            if not album:
                raise HTTPException(404, "álbum não encontrado")
            if album["status"] in ("PAGO", "EXPIRADO", "CANCELADO"):
                raise HTTPException(409, f"álbum já {album['status']}, crie um novo")
            if posicao < 0 or posicao > album["qtd_fotos"]:
                raise HTTPException(400, "posicao fora do range do pack")

            # salva em disco (normaliza pra PNG)
            normalized = _normalize_for_openai(raw)
            path = _album_dir(token) / "originals" / f"{posicao:02d}.png"
            path.write_bytes(normalized)

            cur.execute(
                """
                INSERT INTO colorir.fotos (album_id, posicao, eh_capa, original_path, status)
                VALUES (%s, %s, %s, %s, 'UPLOADED')
                ON CONFLICT (album_id, posicao) DO UPDATE
                  SET eh_capa = EXCLUDED.eh_capa,
                      original_path = EXCLUDED.original_path,
                      status = 'UPLOADED',
                      processada_path = NULL,
                      erro_msg = NULL
                """,
                (album["id"], posicao, eh_capa, str(path)),
            )
            cur.execute(
                "UPDATE colorir.albuns SET status='AGUARDA_ENVIO', atualizado_em=NOW() WHERE id=%s",
                (album["id"],),
            )

    _record_event(album["id"], None, "FOTO_UPLOAD", {"posicao": posicao, "eh_capa": eh_capa})
    return {"ok": True, "posicao": posicao}


@router.get("/album/{token}/foto/{posicao}")
def get_foto(token: str, posicao: int):
    """Serve a foto original (PNG normalizado) — pra mostrar thumb após reload."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.original_path
                  FROM colorir.fotos f
                  JOIN colorir.albuns a ON a.id = f.album_id
                 WHERE a.token = %s AND f.posicao = %s
                """,
                (token, posicao),
            )
            row = cur.fetchone()
    if not row or not row["original_path"]:
        raise HTTPException(404, "foto não encontrada")
    p = Path(row["original_path"])
    if not p.exists():
        raise HTTPException(404, "arquivo não existe mais")
    return FileResponse(p, media_type="image/png")


@router.delete("/album/{token}/foto/{posicao}")
def delete_foto(token: str, posicao: int):
    """Remove uma foto (libera slot pra reupload)."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.id, f.original_path, f.processada_path, a.id AS album_id, a.status
                  FROM colorir.fotos f
                  JOIN colorir.albuns a ON a.id = f.album_id
                 WHERE a.token = %s AND f.posicao = %s
                """,
                (token, posicao),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "foto não encontrada")
            if row["status"] in ("PAGO", "PROCESSANDO", "PREVIEW"):
                raise HTTPException(409, f"álbum em estado {row['status']} — não pode editar")
            cur.execute("DELETE FROM colorir.fotos WHERE id=%s", (row["id"],))
            for path_field in (row["original_path"], row["processada_path"]):
                if path_field:
                    try:
                        Path(path_field).unlink(missing_ok=True)
                    except Exception as e:
                        log.warning(f"falha apagar {path_field}: {e}")

    _record_event(row["album_id"], None, "FOTO_DELETED", {"posicao": posicao})
    return {"ok": True, "posicao": posicao}


@router.get("/album/{token}/status", response_model=StatusOut)
def status_album(token: str):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM colorir.albuns WHERE token = %s", (token,))
            album = cur.fetchone()
            if not album:
                raise HTTPException(404, "álbum não encontrado")
            cur.execute(
                "SELECT posicao, eh_capa, status, erro_msg FROM colorir.fotos WHERE album_id = %s ORDER BY posicao",
                (album["id"],),
            )
            fotos = cur.fetchall()
    valor_str = f"R$ {album['valor_centavos']/100:.2f}".replace(".", ",")
    wa_msg = f"Já fiz o pix de {valor_str}, segue comprovante 📎"
    import urllib.parse
    wa_url = f"https://wa.me/{WHATSAPP_NUM}?text={urllib.parse.quote(wa_msg)}"
    return StatusOut(
        token=album["token"],
        status=album["status"],
        qtd_fotos=album["qtd_fotos"],
        fotos_uploaded=len(fotos),
        nome=album["nome"],
        capa_estilo=album["capa_estilo"],
        valor_centavos=album["valor_centavos"],
        pdf_preview_url=album["pdf_preview_url"],
        pdf_final_url=album["pdf_final_url"],
        fotos=[dict(f) for f in fotos],
        pix_chave=PIX_CHAVE,
        whatsapp_url=wa_url,
    )


def _process_album_background(album_id: int, token: str):
    """Roda em background: chama IA pra cada foto + monta PDF preview."""
    log.info(f"[{token}] iniciando processamento")
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM colorir.albuns WHERE id = %s", (album_id,))
                album = cur.fetchone()
                cur.execute(
                    "SELECT * FROM colorir.fotos WHERE album_id = %s ORDER BY posicao",
                    (album_id,),
                )
                fotos = cur.fetchall()

        adir = _album_dir(token)

        def proc_one(foto):
            posicao = foto["posicao"]
            eh_capa = foto["eh_capa"]
            try:
                src = Path(foto["original_path"]).read_bytes()
                prompt = PROMPT_PIXAR_CAPA if eh_capa else PROMPT_COLORIR
                size = "1024x1024" if eh_capa else "1024x1536"
                # Capa fica em low (color, OK), miolo em medium (line art precisa precisão)
                quality = "low" if eh_capa else "medium"
                processed = _call_openai_edit(src, prompt, size=size, quality=quality)
                # Vetoriza miolo via potrace pra linhas perfeitas (capa permanece cor original)
                if not eh_capa:
                    processed = _vectorize_lineart(processed, target_w=2048)
                out_path = adir / "processed" / f"{posicao:02d}.png"
                out_path.write_bytes(processed)
                with db_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE colorir.fotos SET status='OK', processada_path=%s, processado_em=NOW() WHERE id=%s",
                            (str(out_path), foto["id"]),
                        )
                return posicao, True, None
            except Exception as e:
                log.exception(f"[{token}] erro foto {posicao}")
                with db_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE colorir.fotos SET status='ERRO', erro_msg=%s WHERE id=%s",
                            (str(e)[:500], foto["id"]),
                        )
                return posicao, False, str(e)

        # Paralelo
        with ThreadPoolExecutor(max_workers=PROCESS_PARALLEL) as pool:
            results = list(pool.map(proc_one, fotos))
        ok_count = sum(1 for _, ok, _ in results if ok)
        log.info(f"[{token}] {ok_count}/{len(results)} fotos OK")

        if ok_count == 0:
            with db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE colorir.albuns SET status='CANCELADO', observacao='Todas fotos falharam' WHERE id=%s",
                        (album_id,),
                    )
            _record_event(album_id, album["phone"], "PROCESSAMENTO_FALHOU", {})
            return

        # Re-fetch fotos do DB pra pegar processada_path/status atualizados
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM colorir.fotos WHERE album_id = %s ORDER BY posicao",
                    (album_id,),
                )
                fotos_fresh = cur.fetchall()

        # Monta PDFs (preview com watermark + final clean)
        pdf_preview = adir / "pdfs" / "preview.pdf"
        pdf_final = adir / "pdfs" / "final.pdf"
        try:
            _build_pdf(album, fotos_fresh, str(pdf_preview), with_watermark=True)
            _build_pdf(album, fotos_fresh, str(pdf_final), with_watermark=False)
        except Exception as e:
            log.exception(f"[{token}] erro montar PDF")
            with db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE colorir.albuns SET status='CANCELADO', observacao=%s WHERE id=%s",
                        (f"Erro PDF: {str(e)[:300]}", album_id),
                    )
            _record_event(album_id, album["phone"], "PDF_FALHOU", {"erro": str(e)[:500]})
            return

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE colorir.albuns
                       SET status='PREVIEW',
                           pdf_preview_url=%s,
                           pdf_final_url=%s,
                           preview_em=NOW(),
                           atualizado_em=NOW()
                     WHERE id=%s
                    """,
                    (str(pdf_preview), str(pdf_final), album_id),
                )
        _record_event(album_id, album["phone"], "PREVIEW_PRONTO", {"ok_count": ok_count})
        log.info(f"[{token}] preview pronto: {pdf_preview}")
    except Exception as e:
        log.exception(f"[{token}] erro processamento álbum")
        # marca como CANCELADO pra desbloquear novo álbum
        try:
            with db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE colorir.albuns SET status='CANCELADO', observacao=%s WHERE id=%s",
                        (f"Erro fatal: {str(e)[:300]}", album_id),
                    )
        except Exception:
            pass
        _record_event(album_id, None, "PROCESSAMENTO_ERRO", {"erro": str(e)[:500]})


def _build_pdf(album: dict, fotos: list, out_path: str, with_watermark: bool):
    """Monta PDF: capa + páginas miolo. with_watermark=True pra preview, False pra final."""
    pages = []
    fotos_sorted = sorted(fotos, key=lambda f: f["posicao"])
    for foto in fotos_sorted:
        if foto["status"] != "OK" or not foto.get("processada_path"):
            continue
        img = Image.open(foto["processada_path"]).convert("RGB")
        if foto["eh_capa"]:
            page = _build_capa_page(album, img)
        else:
            page = _build_miolo_page(img, with_border=True)
        if with_watermark:
            page = _apply_watermark(page, valor=album["valor_centavos"])
        pages.append(page)
    if not pages:
        raise RuntimeError("nenhuma página válida pra PDF")
    pages[0].save(out_path, "PDF", resolution=300.0, save_all=True, append_images=pages[1:])


def _build_miolo_page(line_art: Image.Image, with_border: bool = True) -> Image.Image:
    """Página A4 com a arte centralizada e moldura."""
    big = line_art.resize((line_art.width * 4, line_art.height * 4), Image.LANCZOS)
    page = Image.new("RGB", A4_PX, "white")
    if with_border:
        framed = _add_border(big)
    else:
        framed = big
    aspect = framed.width / framed.height
    avail = (A4_PX[0] - 300, A4_PX[1] - 300)
    if avail[0] / avail[1] > aspect:
        h = avail[1]
        w = int(h * aspect)
    else:
        w = avail[0]
        h = int(w / aspect)
    fit = framed.resize((w, h), Image.LANCZOS)
    page.paste(fit, ((A4_PX[0] - w) // 2, (A4_PX[1] - h) // 2))
    return page


def _add_border(art: Image.Image, pad: int = 80, border: int = 10, radius: int = 70) -> Image.Image:
    inner = 60
    fw = art.width + (pad + inner) * 2
    fh = art.height + (pad + inner) * 2
    framed = Image.new("RGB", (fw, fh), "white")
    framed.paste(art, (pad + inner, pad + inner))
    d = ImageDraw.Draw(framed)
    d.rounded_rectangle(
        [pad, pad, fw - pad, fh - pad],
        radius=radius,
        outline=(40, 40, 40),
        width=border,
    )
    return framed


def _build_capa_page(album: dict, pixar_img: Image.Image) -> Image.Image:
    """Capa = template estilo + foto Pixar + nome."""
    estilo = album.get("capa_estilo") or "familia"
    paletas = {
        "rosa": ("#FFE5EE", "#FFB6C9", "#7B1F3F"),
        "azul": ("#E0F0FF", "#A0CFFF", "#1F3F7B"),
        "familia": ("#FFF4D6", "#FFD37A", "#7B5A1F"),
    }
    bg, accent, dark = paletas.get(estilo, paletas["familia"])
    page = Image.new("RGB", A4_PX, bg)
    d = ImageDraw.Draw(page)

    # Título topo (sem decorações)
    f_top = _font_bold(80)
    title = "MEU LIVRO PRA COLORIR"
    bbox = d.textbbox((0, 0), title, font=f_top)
    d.text(((A4_PX[0] - (bbox[2] - bbox[0])) // 2, 600), title, fill=dark, font=f_top)

    # Foto Pixar centralizada (1700x1700)
    PHOTO_W = 1700
    px = (A4_PX[0] - PHOTO_W) // 2
    py = 950
    d.rounded_rectangle(
        [px - 30, py - 30, px + PHOTO_W + 30, py + PHOTO_W + 30],
        radius=60,
        fill="white",
        outline=dark,
        width=8,
    )
    pixar_resized = pixar_img.resize((PHOTO_W, PHOTO_W), Image.LANCZOS)
    page.paste(pixar_resized, (px, py))

    # Nome embaixo
    nome = (album.get("nome") or "ALBUM DA FAMÍLIA").upper()
    f_name = _font_bold(140)
    bbox = d.textbbox((0, 0), nome, font=f_name)
    text_y = py + PHOTO_W + 100
    d.text(((A4_PX[0] - (bbox[2] - bbox[0])) // 2, text_y), nome, fill=dark, font=f_name)
    return page


def _apply_watermark(page: Image.Image, valor: int) -> Image.Image:
    """Watermark agressivo cobrindo a página inteira."""
    canvas = page.copy()
    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(overlay)
    valor_str = f"R$ {valor / 100:.2f}".replace(".", ",")
    wm_text = f"PREVIEW · PAGUE {valor_str}"
    f = _font_bold(120)
    for y in range(-300, canvas.height + 300, 280):
        for x in range(-400, canvas.width + 400, 1100):
            d.text((x, y), wm_text, font=f, fill=(220, 30, 80, 110))
    rot = overlay.rotate(-22, resample=Image.BICUBIC, expand=False)

    # stamp central PREVIEW gigante
    stamp = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    sd = ImageDraw.Draw(stamp)
    sf = _font_bold(420)
    bbox = sd.textbbox((0, 0), "PREVIEW", font=sf)
    cx = canvas.width // 2 - (bbox[2] - bbox[0]) // 2
    cy = canvas.height // 2 - (bbox[3] - bbox[1]) // 2
    sd.text((cx, cy), "PREVIEW", font=sf, fill=(220, 30, 80, 200))
    stamp_rot = stamp.rotate(-22, resample=Image.BICUBIC, expand=False)

    out = Image.alpha_composite(canvas.convert("RGBA"), rot)
    out = Image.alpha_composite(out, stamp_rot)
    return out.convert("RGB")


def _font_bold(size: int) -> ImageFont.FreeTypeFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


@router.post("/album/{token}/processar")
def processar_album(token: str, body: ProcessarAlbumIn, bg: BackgroundTasks):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, status, phone, qtd_fotos FROM colorir.albuns WHERE token=%s", (token,))
            album = cur.fetchone()
            if not album:
                raise HTTPException(404, "álbum não encontrado")
            if album["status"] in ("PROCESSANDO", "PREVIEW", "PAGO"):
                return {"ok": True, "status": album["status"], "msg": "já em andamento"}

            cur.execute("SELECT COUNT(*) AS c FROM colorir.fotos WHERE album_id=%s", (album["id"],))
            n = cur.fetchone()["c"]
            if n == 0:
                raise HTTPException(400, "nenhuma foto enviada")

            updates = []
            params = []
            if body.nome:
                updates.append("nome=%s")
                params.append(body.nome)
            if body.capa_estilo:
                updates.append("capa_estilo=%s")
                params.append(body.capa_estilo)
            updates.append("status='PROCESSANDO'")
            updates.append("enviado_em=NOW()")
            updates.append("atualizado_em=NOW()")
            params.append(album["id"])
            cur.execute(f"UPDATE colorir.albuns SET {', '.join(updates)} WHERE id=%s", params)

    _record_event(album["id"], album["phone"], "CLICOU_ENVIAR", {"qtd": n})
    bg.add_task(_process_album_background, album["id"], token)
    return {"ok": True, "status": "PROCESSANDO", "fotos": n}


@router.get("/album/{token}/preview")
def get_preview(token: str):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pdf_preview_url, status FROM colorir.albuns WHERE token=%s", (token,)
            )
            row = cur.fetchone()
    if not row or not row["pdf_preview_url"]:
        raise HTTPException(404, "preview não disponível")
    return FileResponse(row["pdf_preview_url"], media_type="application/pdf", filename="preview.pdf")


@router.get("/album/{token}/preview-image")
def get_preview_image(token: str):
    """Renderiza a capa preview como PNG (com watermark) — funciona em qualquer browser
    que não consegue renderizar iframe PDF (iOS Safari antigo, etc)."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM colorir.albuns WHERE token=%s", (token,))
            album = cur.fetchone()
            if not album or album["status"] not in ("PREVIEW",):
                raise HTTPException(404, "preview não disponível")
            cur.execute(
                "SELECT * FROM colorir.fotos WHERE album_id=%s AND eh_capa=true LIMIT 1",
                (album["id"],),
            )
            capa = cur.fetchone()

    # Se não tem capa, usa primeira página processada
    pic_path = None
    if capa and capa["processada_path"]:
        pic_path = capa["processada_path"]
    else:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT processada_path FROM colorir.fotos WHERE album_id=%s AND status='OK' ORDER BY posicao LIMIT 1",
                    (album["id"],),
                )
                r = cur.fetchone()
                if r:
                    pic_path = r["processada_path"]

    if not pic_path:
        raise HTTPException(404, "imagem não encontrada")

    img = Image.open(pic_path).convert("RGB")
    if capa and capa["eh_capa"]:
        # capa Pixar — compõe como capa do livro
        page = _build_capa_page(album, img)
    else:
        # se for miolo, usa direto
        page = _build_miolo_page(img, with_border=True)
    page = _apply_watermark(page, valor=album["valor_centavos"])

    # Reduz tamanho pra browser carregar rápido (1200px largura ~suficiente pra preview)
    if page.width > 1200:
        ratio = 1200 / page.width
        page = page.resize((1200, int(page.height * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    page.save(buf, "JPEG", quality=85, optimize=True)
    buf.seek(0)

    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@router.post("/album/{token}/liberar")
def liberar_album(token: str, body: LiberarIn):
    """Chamado pelo n8n após PIX validado. Move pra histórico, deleta originais."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM colorir.albuns WHERE token=%s", (token,))
            album = cur.fetchone()
            if not album:
                raise HTTPException(404, "álbum não encontrado")
            if album["status"] == "PAGO":
                return {"ok": True, "status": "PAGO", "pdf_url": album["pdf_final_url"]}
            if not album["pdf_final_url"]:
                raise HTTPException(409, "PDF final ainda não gerado")

            cur.execute(
                """
                UPDATE colorir.albuns
                   SET status='PAGO',
                       pago_em=NOW(),
                       atualizado_em=NOW(),
                       observacao=COALESCE(observacao,'') || %s
                 WHERE id=%s
                """,
                (f" pix={body.msg_id_comprovante or ''}", album["id"]),
            )
            # arquiva no histórico
            cur.execute(
                """
                INSERT INTO colorir.pdfs_entregues (album_id, phone, nome, capa_estilo, qtd_fotos, valor_pago, pdf_url)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    album["id"],
                    album["phone"],
                    album["nome"],
                    album["capa_estilo"],
                    album["qtd_fotos"],
                    body.valor_pago or album["valor_centavos"],
                    album["pdf_final_url"],
                ),
            )

    # apaga originais (privacidade)
    try:
        adir = STORAGE_ROOT / token / "originals"
        if adir.exists():
            for f in adir.iterdir():
                f.unlink()
    except Exception as e:
        log.warning(f"falha apagar originals: {e}")

    _record_event(album["id"], album["phone"], "PIX_VALIDADO", {"valor": body.valor_pago})
    return {"ok": True, "status": "PAGO", "pdf_url": album["pdf_final_url"]}


@router.get("/album/{token}/pdf-final")
def get_pdf_final(token: str):
    """Cliente baixa o PDF clean depois de pago."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pdf_final_url, status FROM colorir.albuns WHERE token=%s", (token,)
            )
            row = cur.fetchone()
    if not row or row["status"] != "PAGO":
        raise HTTPException(403, "álbum não pago")
    return FileResponse(row["pdf_final_url"], media_type="application/pdf", filename="album.pdf")


@router.get("/exemplo/{kind}")
def get_exemplo(kind: str):
    """Serve exemplos before/after/audio (configurados em /data/colorir/exemplos/).
    kind: 'antes' | 'depois' | 'audio'"""
    safe = {"antes": "antes.jpg", "depois": "depois.jpg", "audio": "intro.mp3"}
    if kind not in safe:
        raise HTTPException(404, "kind inválido")
    p = STORAGE_ROOT / "exemplos" / safe[kind]
    if not p.exists():
        raise HTTPException(404, "arquivo não disponível")
    media = "image/jpeg" if kind != "audio" else "audio/mpeg"
    return FileResponse(p, media_type=media)


@router.get("/historico/{phone}")
def historico(phone: str):
    """Lista PDFs pagos pra 'Meus Livros'."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, album_id, nome, capa_estilo, qtd_fotos, valor_pago, pago_em, baixado_count
                  FROM colorir.pdfs_entregues
                 WHERE phone = %s
                 ORDER BY pago_em DESC
                """,
                (phone,),
            )
            rows = cur.fetchall()
    return {"ok": True, "total": len(rows), "items": [dict(r) for r in rows]}
