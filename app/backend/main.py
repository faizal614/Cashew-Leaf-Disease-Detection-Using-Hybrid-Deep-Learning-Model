from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Cashew anthracnose",
    "Cashew gummosis",
    "Cashew healthy",
    "Cashew leaf miner",
    "Cashew red rust"
]

PRESCRIPTIONS = {
    "Cashew anthracnose": {
        "cause": "Fungal infection (Colletotrichum species)",
        "treatment": [
            "Spray Carbendazim (0.1%) or Mancozeb (0.2%)",
            "Apply Copper oxychloride (0.3%)",
            "Remove infected leaves"
        ],
        "prevention": [
            "Ensure proper spacing",
            "Avoid overhead irrigation",
            "Regular orchard sanitation"
        ]
    },
    "Cashew gummosis": {
        "cause": "Fungal infection or trunk injury",
        "treatment": [
            "Scrape affected bark",
            "Apply Bordeaux paste (1%)",
            "Spray Metalaxyl + Mancozeb"
        ],
        "prevention": [
            "Avoid trunk injury",
            "Ensure drainage",
            "Monitor during monsoon"
        ]
    },
    "Cashew leaf miner": {
        "cause": "Leaf mining insect",
        "treatment": [
            "Spray Imidacloprid (0.3 ml/L)",
            "Use Neem oil (3%)",
            "Remove infested leaves"
        ],
        "prevention": [
            "Encourage natural predators",
            "Regular monitoring",
            "Avoid excess nitrogen"
        ]
    },
    "Cashew red rust": {
        "cause": "Algal infection",
        "treatment": [
            "Spray Copper oxychloride (0.3%)",
            "Apply Bordeaux mixture (1%)",
            "Prune infected branches"
        ],
        "prevention": [
            "Improve sunlight exposure",
            "Maintain spacing",
            "Control humidity"
        ]
    },
    "Cashew healthy": {
        "cause": "No disease detected",
        "treatment": ["No treatment required"],
        "prevention": [
            "Maintain fertilization",
            "Proper irrigation",
            "Periodic inspection"
        ]
    }
}

vit_model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=5
)

vit_model.load_state_dict(
    torch.load("best_vit_model copy.pth", map_location=device)
)

vit_model.to(device)
vit_model.eval()

efficientnet_model = timm.create_model(
    "efficientnet_b4",
    pretrained=False,
    num_classes=5
)

efficientnet_model.load_state_dict(
    torch.load("best_efficientnet_model.pth", map_location=device)
)

efficientnet_model.to(device)
efficientnet_model.eval()

class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.eff = timm.create_model(
            "efficientnet_b4",
            pretrained=False
        )
        self.eff.classifier = nn.Identity()

        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False
        )
        self.vit.head = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(1792 + 192, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, eff_img, vit_img):
        eff_feat = self.eff(eff_img)
        vit_feat = self.vit(vit_img)

        combined = torch.cat((eff_feat, vit_feat), dim=1)
        return self.fc(combined)

hybrid_model = HybridModel(num_classes=5)

hybrid_model.load_state_dict(
    torch.load("best_hybrid_model_b4.pth", map_location=device)
)

hybrid_model.to(device)
hybrid_model.eval()

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

@app.get("/")
def root():
    return {"message": "Cashew Leaf Disease Detection API is running"}

@app.post("/predict/vit")
async def predict_vit(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = vit_model(image)
        probs = torch.softmax(outputs, dim=1)

        print("VIT probs:", probs)
        print("VIT predicted index:", torch.argmax(probs, dim=1).item())

        conf, pred = torch.max(probs, 1)

    return format_response(
        "ViT Base Patch16 224",
        pred,
        conf,
        CLASS_NAMES
    )

@app.post("/predict/efficientnet")
async def predict_efficientnet(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = efficientnet_model(image)
        probs = torch.softmax(outputs, dim=1)

        print("EFF probs:", probs)
        print("EFF predicted index:", torch.argmax(probs, dim=1).item())

        conf, pred = torch.max(probs, 1)

    return format_response(
        "EfficientNet-B4",
        pred,
        conf,
        CLASS_NAMES
    )

@app.post("/predict/hybrid")
async def predict_hybrid(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    eff_image = inference_transform(image).unsqueeze(0).to(device)
    vit_image = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = hybrid_model(eff_image, vit_image)
        probs = torch.softmax(outputs, dim=1)

        print("HYBRID probs:", probs)
        print("HYBRID predicted index:", torch.argmax(probs, dim=1).item())

        conf, pred = torch.max(probs, 1)

    return format_response(
        "Hybrid (EfficientNet-B4 + ViT-Tiny)",
        pred,
        conf,
        CLASS_NAMES
    )

def format_response(model_name, pred, conf, class_names):
    label = class_names[pred.item()]

    return {
        "model": model_name,
        "prediction": label,
        "status": "Healthy" if label == "Cashew healthy" else "Diseased",
        "confidence": round(conf.item() * 100, 2),
        "prescription": PRESCRIPTIONS[label]
    }
