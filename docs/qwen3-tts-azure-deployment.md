# Qwen3-TTS Deployment on Azure GPU VM

## Overview
Deploy Qwen3-TTS (open-source text-to-speech model) on Azure GPU Virtual Machine using Azure free credits.

---

## Prerequisites
- Azure account with free credits ($200 for new users)
- SSH client (Terminal on Mac/Linux, or PuTTY on Windows)

---

## Step 1: Create Azure GPU Virtual Machine

### 1.1 Login to Azure Portal
```
https://portal.azure.com
```

### 1.2 Create Virtual Machine
1. Click **"Create a resource"** → **"Virtual Machine"**
2. Configure basics:
   - **Subscription**: Select your subscription with free credits
   - **Resource group**: Create new → `qwen-tts-rg`
   - **VM name**: `qwen-tts-vm`
   - **Region**: `Southeast Asia` (or nearest region with GPU availability)
   - **Image**: `Ubuntu Server 22.04 LTS - x64 Gen2`
   - **Size**: Click "See all sizes" → Filter by GPU:
     - **Recommended**: `Standard_NC4as_T4_v3` (1x T4 GPU, 16GB VRAM, ~$0.53/hr)
     - **Alternative**: `Standard_NC6s_v3` (1x V100 GPU, 16GB VRAM, ~$3.06/hr)

### 1.3 Configure Authentication
- **Authentication type**: SSH public key
- **Username**: `azureuser`
- **SSH public key source**: Generate new key pair
- Download and save the private key (.pem file)

### 1.4 Configure Networking
- Allow inbound ports: **SSH (22)**
- Add port **8000** for API access

### 1.5 Create VM
Click **"Review + create"** → **"Create"**

---

## Step 2: Connect to VM

### 2.1 Get VM Public IP
- Go to your VM in Azure Portal
- Copy the **Public IP address**

### 2.2 SSH Connect
```bash
chmod 400 ~/Downloads/qwen-tts-vm_key.pem
ssh -i ~/Downloads/qwen-tts-vm_key.pem azureuser@<YOUR_VM_IP>
```

---

## Step 3: Install NVIDIA Drivers & CUDA

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, verify:
```bash
nvidia-smi
```

---

## Step 4: Install Python Environment

```bash
sudo apt install -y python3.10 python3.10-venv python3-pip
python3 -m venv ~/qwen-tts-env
source ~/qwen-tts-env/bin/activate
pip install --upgrade pip
```

---

## Step 5: Install Qwen3-TTS

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate soundfile
pip install qwen-tts
pip install huggingface_hub[cli]
```

---

## Step 6: Download Model

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./qwen3-tts-model
```

---

## Step 7: Create API Server

Save as `tts_server.py`:
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
import io
import time

app = FastAPI(title="Qwen3-TTS API")

model_path = "./qwen3-tts-model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

class TTSRequest(BaseModel):
    text: str
    voice: str = "default"
    language: str = "en"

@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    start_time = time.time()
    inputs = tokenizer(request.text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048)
    
    audio = outputs[0].cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format='WAV')
    buffer.seek(0)
    
    return Response(
        content=buffer.read(),
        media_type="audio/wav",
        headers={
            "X-Latency-Ms": str((time.time() - start_time) * 1000),
            "X-Duration-Seconds": str(len(audio) / 24000),
        }
    )

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Run:
```bash
pip install fastapi uvicorn
uvicorn tts_server:app --host 0.0.0.0 --port 8000
```

---

## Cost Estimation

| Resource | Cost |
|----------|------|
| NC4as_T4_v3 VM | ~$0.53/hour |
| Storage (128GB) | ~$5/month |
| **Monthly (8 hrs/day)** | **~$127** |

**Save costs:** Deallocate when not in use:
```bash
az vm deallocate --name qwen-tts-vm --resource-group qwen-tts-rg
```
