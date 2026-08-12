import os
import shutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from services.stt_service.stt_service import transcribe_audio_file

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

app = FastAPI()

@app.websocket("/ws/transcribe")
async def websocket_realtime_transcribe(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for Speech-to-Text streaming.
    Receives raw audio blobs from frontend, buffers them, and streams back text live.
    """
    await websocket.accept()
    print("Real-time STT client connected via WebSocket")
    
    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            if len(audio_buffer) > 32000:
                temp_filename = f"temp_live_{os.getpid()}.webm"
                
                with open(temp_filename, "wb") as f:
                    f.write(audio_buffer)
                
                audio_buffer.clear()
                
                transcript = transcribe_audio_file(temp_filename)
                
                if transcript:
                    await websocket.send_json({
                        "status": "success",
                        "transcript": transcript
                    })
                
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    
    except WebSocketDisconnect:
        print("Real-time STT client disconnected")
    except Exception as e:
        print(f"WebSocket STT Error: {e}")
        await websocket.close()

@app.post("/transcribe")
async def http_transcribe(audio: UploadFile = File(...)):
    """
    HTTP POST endpoint for one-shot Speech-to-Text.
    Receives an uploaded audio file, transcribes it, and returns the text.
    """
    temp_filename = f"temp_upload_{os.getpid()}_{audio.filename}"
    try:
        # Save uploaded file
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # Transcribe
        transcript = transcribe_audio_file(temp_filename)
        
        return {
            "status": "ok",
            "language": "id",
            "text": transcript,
            "message": "Transcription successful" if transcript else "No speech detected"
        }
    except Exception as e:
        print(f"HTTP Transcribe Error: {e}")
        return {
            "status": "error",
            "language": "id",
            "text": "",
            "message": str(e)
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)