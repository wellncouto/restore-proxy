# Restore Proxy

Mini-proxy HTTP que conversa com o HF Space `titanito/Image_Face_Upscale_Restoration-GFPGAN-RestoreFormer-CodeFormer-GPEN` via gradio_client.

Recebe foto em base64, devolve foto restaurada em base64. Zero custo de API, ~150 MB de RAM.

## Endpoints

### `GET /health`
Liveness check.

### `POST /restore`
Body:
```json
{
  "image_b64": "<base64 da foto, com ou sem prefixo data:>",
  "face_model": "GFPGANv1.4.pth",                 // opcional
  "upscale_model": "SRVGG, realesr-general-x4v3.pth", // opcional
  "scale": 2                                       // opcional
}
```

Retorna:
```json
{
  "image_b64": "<base64 da foto restaurada>",
  "mime_type": "image/png",
  "face_model": "GFPGANv1.4.pth",
  "upscale_model": "SRVGG, realesr-general-x4v3.pth",
  "scale": 2
}
```

## Modelos disponíveis (face_model)

- `GFPGANv1.4.pth` (default — bom geral)
- `RestoreFormer++.ckpt` (melhor em rosto severo)
- `CodeFormer.pth` (top em foto muito danificada)
- `GPEN-BFR-512.pth` / `GPEN-BFR-1024.pt` / `GPEN-BFR-2048.pt`
- `GFPGANv1.3.pth` / `GFPGANv1.2.pth`
- `RestoreFormer.ckpt`

## Deploy no Easypanel

1. Cria novo serviço **App** → tipo **Dockerfile**
2. Source: upload do zip OU aponta pra Git
3. Porta interna: `8000`
4. Nome interno do container: `restore-proxy` (ou outro)
5. Não precisa expor publicamente — n8n acessa pela rede interna
6. Variáveis de ambiente (opcional):
   - `HF_SPACE` (padrão: titanito)
   - `FACE_MODEL` (padrão: GFPGANv1.4.pth)
   - `UPSCALE_MODEL` (padrão: SRVGG, realesr-general-x4v3.pth)
   - `SCALE` (padrão: 2)
   - `HF_TOKEN` (opcional, só se Space pedir auth no futuro)

Após deploy o n8n acessa em `http://restore-proxy:8000/restore` (ou o nome que você der ao container).

## Teste local

```bash
docker build -t restore-proxy .
docker run -p 8000:8000 restore-proxy

# em outro terminal
curl -X POST http://localhost:8000/restore \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\":\"$(base64 -i foto.jpg)\"}" \
  | jq -r .image_b64 | base64 -d > restaurada.png
```
