import torch
from PIL import Image
from transformers import DistilBertTokenizer
from torchvision import transforms
from CLIP import CLIPModel

# CLIPModel, ImageEncoder, TextEncoder 클래스가 정의되어 있다고 가정

# 모델 및 토크나이저 인스턴스 생성
model = CLIPModel()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 저장된 모델 파라미터 로드 (strict=False 옵션으로 누락 키 무시)
model.load_state_dict(torch.load("best_model_v2.pth", map_location=device), strict=False)
model.to(device)
model.eval()

# 분석할 이미지 파일 경로와 여러 텍스트 정의
image_path = "zero-shot/cat.jpg"
texts = [
    "a photo of a airplane",
    "a photo of a automobile",
    "a photo of a bird",
    "a photo of a cat",
    "a photo of a deer",
    "a photo of a dog",
    "a photo of a frog",
    "a photo of a horse",
    "a photo of a ship",
    "a photo of a truck"
]

# 텍스트 토크나이즈: 배치 형태로 변환 (padding, truncation 적용)
text_inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
for key in text_inputs:
    text_inputs[key] = text_inputs[key].to(device)

# 이미지 전처리: 이미지 로드 후 모델에 맞게 resize, tensor 변환, normalization
img = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),    # 모델 입력에 맞게 크기 조정 (예시)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 일반적인 ImageNet normalization
])
img_tensor = transform(img).unsqueeze(0).to(device)  # 배치 차원 추가

# 추론: image_encoder와 text_encoder를 통해 각각 임베딩을 구한 후 logit 계산
with torch.no_grad():
    image_embedding = model.image_encoder(img_tensor)  # shape: (1, proj_dim)
    text_embedding = model.text_encoder(text_inputs["input_ids"], text_inputs["attention_mask"])  # shape: (num_texts, proj_dim)
    
    # logit 계산: 이미지 임베딩과 텍스트 임베딩의 내적에 temperature 스케일 적용
    logits = image_embedding @ text_embedding.T * torch.exp(model.temperature)
    # logits의 shape: (1, num_texts) -> squeeze해서 (num_texts,)로 변환
    logits = logits.squeeze(0)

# 가장 높은 logit 값을 가진 텍스트 인덱스 추출
pred_index = torch.argmax(logits).item()

# 결과 출력
print("Logits:", logits)
print("Predicted text:", texts[pred_index])
