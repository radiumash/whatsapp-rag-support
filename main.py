import os
from dotenv import load_dotenv

load_dotenv()
import requests
from fastapi import FastAPI, Request, Response, Query, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import redis
from rag_engine import generate_rag_response, ingest_any_source

app = FastAPI(title="Support RAG Assistant")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Redis session setup
in_memory_sessions = {}
redis_client = None

try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
        socket_connect_timeout=1
    )
    r.ping()
    redis_client = r
except Exception:
    pass

def get_history(key: str) -> str:
    if redis_client:
        try:
            return redis_client.get(key) or ""
        except Exception:
            pass
    return in_memory_sessions.get(key, "")

def set_history(key: str, value: str):
    if redis_client:
        try:
            redis_client.set(key, value, ex=3600)
            return
        except Exception:
            pass
    in_memory_sessions[key] = value

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

@app.get("/playground")
async def serve_playground():
    return FileResponse("static/playground.html")

MAX_FILE_SIZE = 8 * 1024 * 1024  # 8 MB

# 1. Multi-File Ingestion Endpoint (.pdf, .xlsx, .xls, .csv, .txt, .md)
@app.post("/api/upload_any")
async def upload_any_file(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    allowed_exts = [".pdf", ".xlsx", ".xls", ".csv", ".txt", ".md"]
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Allowed: {', '.join(allowed_exts)}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds maximum allowed size of 8 MB.")

    try:
        result = ingest_any_source(
            source_type="file",
            file_bytes=file_bytes,
            filename=file.filename,
            session_id=session_id
        )
        return {
            "status": "success",
            "chunks_count": result["count"],
            "chunks": result["chunks"],
            "source_name": result["source_name"],
            "conversation_starter": result["conversation_starter"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

# 2. Web URL Ingestion Endpoint
class UrlPayload(BaseModel):
    url: str
    session_id: str

@app.post("/api/index_url")
async def index_web_url(payload: UrlPayload):
    if not payload.url.startswith("http://") and not payload.url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Please enter a valid URL starting with http:// or https://")

    try:
        result = ingest_any_source(
            source_type="url",
            url=payload.url,
            session_id=payload.session_id
        )
        return {
            "status": "success",
            "chunks_count": result["count"],
            "chunks": result["chunks"],
            "source_name": result["source_name"],
            "conversation_starter": result["conversation_starter"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape & index URL: {str(e)}")

# 3. Playground Chat Route
class ChatPayload(BaseModel):
    message: str
    session_id: str

@app.post("/api/playground_chat")
async def playground_chat(payload: ChatPayload):
    history_key = f"playground:{payload.session_id}"
    history = get_history(history_key)

    bot_reply = generate_rag_response(
        query=payload.message,
        history=history,
        namespace=payload.session_id
    )

    updated_history = f"{history}\nUser: {payload.message}\nBot: {bot_reply}"[-8000:]
    set_history(history_key, updated_history)

    return {"reply": bot_reply}

@app.post("/api/chat")
async def web_chat(payload: ChatPayload):
    history_key = f"web_chat:{payload.session_id}"
    history = get_history(history_key)
    bot_reply = generate_rag_response(payload.message, history, payload.session_id)
    updated_history = f"{history}\nUser: {payload.message}\nBot: {bot_reply}"[-8000:]
    set_history(history_key, updated_history)
    return {"reply": bot_reply}

# ----------------------------------------------------
# 2. WhatsApp Cloud API Endpoints
# ----------------------------------------------------
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN")
@app.get("/webhook")
async def verify_webhook(
    mode: str = Query(..., alias="hub.mode"),
    token: str = Query(..., alias="hub.verify_token"),
    challenge: str = Query(..., alias="hub.challenge")
):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return Response(content=challenge, media_type="text/plain")
    return Response(status_code=403)

@app.post("/webhook")
async def receive_whatsapp(request: Request):
    data = await request.json()
    try:
        entry = data["entry"][0]["changes"][0]["value"]
        if "messages" in entry:
            message = entry["messages"][0]
            from_number = message["from"]
            
            if message["type"] == "text":
                msg_body = message["text"]["body"]
                history_key = f"wa_chat:{from_number}"
                history = redis_client.get(history_key) or ""

                # Generate Answer
                bot_reply = generate_rag_response(msg_body, history, from_number)

                # Update Redis
                updated_history = f"{history}\nUser: {msg_body}\nBot: {bot_reply}"[-8000:]
                redis_client.set(history_key, updated_history, ex=3600)

                # Send WhatsApp reply
                send_whatsapp_message(from_number, bot_reply)
    except Exception as e:
        print(f"Webhook processing error: {e}")

    return {"status": "success"}

def send_whatsapp_message(recipient_id: str, text: str):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_id,
        "type": "text",
        "text": {"body": text}
    }
    requests.post(url, json=payload, headers=headers)