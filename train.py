import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms
from transformers import DistilBertTokenizer
from models import CLIPModel  # models.py 내의 CLIPModel
from PIL import Image
from tqdm import tqdm

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 초기화 (text_encoder.py에서 사용한 것과 동일)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 이미지 전처리: ViT의 입력 사이즈(224x224)에 맞게 변환
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def collate_fn(batch):
    images = []
    texts = []
    for item in batch:
        # item["image"]는 PIL.Image 객체 또는 numpy 배열일 수 있으므로 PIL.Image로 변환
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        images.append(image_transform(img))
        # 캡션은 리스트 형태일 수 있으므로 첫번째 항목 사용
        caption = item["caption"][0] if isinstance(item["caption"], list) else item["caption"]
        texts.append(caption)
    images = torch.stack(images, dim=0)
    # 텍스트 토큰화 (max_length는 상황에 따라 조정)
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=40)
    # text_encoder.py에서 forward 함수는 (input_ids, attention_mask)를 받지만,
    # models.py에서는 "mask" 키를 사용하므로 키 이름을 변경합니다.
    tokenized = {**tokenized, "mask": tokenized.pop("attention_mask")}
    return {"image": images, "input_ids": tokenized["input_ids"], "mask": tokenized["mask"]}

def train(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        # 데이터를 device로 이동
        batch["image"] = batch["image"].to(device)
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["mask"] = batch["mask"].to(device)
        
        optimizer.zero_grad()
        loss, _ = model(batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch["image"] = batch["image"].to(device)
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["mask"] = batch["mask"].to(device)
            
            loss, _ = model(batch)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def main():
    # COCO 캡션 데이터셋의 1000개 샘플을 불러옵니다.
    dataset = load_dataset("coco_captions", "2017", split="train[:1000]")
    
    # 6:2:2 비율로 train/validation/test 분리
    split_1 = dataset.train_test_split(test_size=0.4, seed=42)
    train_dataset = split_1["train"]       # 약 60% (600개 샘플)
    rest_dataset = split_1["test"]           # 약 40% (400개 샘플)
    split_2 = rest_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = split_2["train"]           # 약 20% (200개 샘플)
    test_dataset = split_2["test"]           # 약 20% (200개 샘플)
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # 모델 초기화 및 옵티마이저 설정
    model = CLIPModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} \t Validation Loss: {val_loss:.4f}")
    
    # 학습 완료 후 test 데이터셋으로 최종 평가
    test_loss = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    # 학습된 모델 저장
    torch.save(model.state_dict(), "clip_model.pth")
    print("Training complete and model saved as clip_model.pth")

if __name__ == "__main__":
    main()