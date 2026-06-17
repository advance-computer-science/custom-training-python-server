import io
import base64
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
import uvicorn


# MODEL ARCHITECTURE

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x): 
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bn = self.bottleneck(self.pool4(d4))

        u1 = self.conv_up1(torch.cat([self.up1(bn), d4], dim=1))
        u2 = self.conv_up2(torch.cat([self.up2(u1), d3], dim=1))
        u3 = self.conv_up3(torch.cat([self.up3(u2), d2], dim=1))
        u4 = self.conv_up4(torch.cat([self.up4(u3), d1], dim=1))
        return self.out(u4)

# SERVER & MODEL INITIALIZATION
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Update if your React port is different
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model into memory at while creating server.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Starting server. Loading U-Net onto: {device}")

model = UNet(in_channels=3, out_channels=1)
WEIGHTS_FILE = "unet_lgg_weights.pth" # Ensure this path is correct
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# THE API ENDPOINT

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # read and process Image
        contents = await file.read()
        raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(raw_image).unsqueeze(0).to(device)

        # run inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            prediction_mask = (probs > 0.5).float() # Returns 1.0 for tumor, 0.0 for background

        # convert Tensor to Base64 Image
        mask_2d = prediction_mask.squeeze().cpu().numpy() 
        
        mask_uint8 = (mask_2d * 255).astype(np.uint8) 
        
        # create PIL image from numpy array
        mask_image = Image.fromarray(mask_uint8, mode="L")
        
        # save image to a byte buffer in PNG format
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        
        # encode the bytes into a base64 string
        base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success", 
            "mask_base64": base64_string,
            "message": "Inference complete."
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("Running U-Net Inference API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)