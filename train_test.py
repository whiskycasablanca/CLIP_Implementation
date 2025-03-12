# device 설정
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from datasets import load_dataset
import torchvision.transforms as transforms
from transformers import DistilBertTokenizer
from models import CLIPModel  # models.py 내의 CLIPModel
from PIL import Image
from tqdm import tqdm
from text_encoder import TextEncoder
import torchvision.transforms as T
from dataset import Flickr8kDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저와 이미지 전처리 정의
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
])

# train split 파일: Flickr_8k.testImages.txt (이미지 파일명만 한 줄씩)
train_dataset = Flickr8kDataset(
    img_folder='images',
    caption_file='captions.txt',
    split_file='Flickr_8k.trainImages.txt',  # 분할 파일 경로 지정
    transform=transform,  # 미리 정의한 transform
    tokenizer=tokenizer ,  # 미리 정의한 토크나이저
    max_length=40
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

# validation split 파일: Flickr_8k.testImages.txt (이미지 파일명만 한 줄씩)
val_dataset = Flickr8kDataset(
    img_folder='images',
    caption_file='captions.txt',
    split_file='Flickr_8k.devImages.txt',  # 분할 파일 경로 지정
    transform=transform,  # 미리 정의한 transform
    tokenizer=tokenizer ,  # 미리 정의한 토크나이저
    max_length=40
)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)



# test split 파일: Flickr_8k.testImages.txt (이미지 파일명만 한 줄씩)
test_dataset = Flickr8kDataset(
    img_folder='images',
    caption_file='captions.txt',
    split_file='Flickr_8k.testImages.txt',  # 분할 파일 경로 지정
    transform=transform,  # 미리 정의한 transform
    tokenizer=tokenizer ,  # 미리 정의한 토크나이저
    max_length=40
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 예시: 간단한 CLIP-like 모델 가정 (직접 구현한 CLIPModel이 있다고 가정)
# from model import CLIPModel  # 이미 구현되어 있다면 import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel().to(device)  # 사용 중인 CLIP 모델 클래스로 교체
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for batch in test_loader:
    print("Image shape:", batch["image"].shape)
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Mask shape:", batch["mask"].shape)
    break  # 첫 배치만 확인 break


num_epochs = 5  # 원하는 에폭 수
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # === 2.1 Training phase ===
    model.train()  # 학습 모드
    train_loss = 0.0
    for batch in train_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        mask = batch["mask"].to(device)

        # 모델에 입력할 딕셔너리 구성 (CLIPModel의 forward가 이를 받도록 설계)
        inputs = {
            "image": images,
            "input_ids": input_ids,
            "mask": mask
        }

        # forward & loss 계산
        loss = model(inputs)

        # 역전파 & 옵티마이저 스텝
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    train_loss /= len(train_loader)

    # === 2.2 Validation phase ===
    model.eval()  # 평가 모드
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)

            inputs = {
                "image": images,
                "input_ids": input_ids,
                "mask": mask
            }

            loss = model(inputs)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # === 2.3 모델 체크포인트 저장(옵션) ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved.")