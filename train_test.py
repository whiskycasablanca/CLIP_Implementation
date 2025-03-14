import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from CLIP import CLIPModel  # models.py 내의 CLIPModel
import torchvision.transforms as T
from dataset import Flickr8kDataset
import matplotlib.pyplot as plt
from tqdm import tqdm


# 토크나이저와 이미지 전처리 정의
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
])

# split된 데이터셋 생성성
split_files = {'train' : 'Flickr_8k.trainImages.txt', 'validation' : 'Flickr_8k.devImages.txt', 'test' : 'Flickr_8k.testImages.txt'}
datasets = {
    split_name : Flickr8kDataset(
        img_folder='images',
        caption_file='captions.txt',
        split_file=split_file,  # 분할 파일 경로 지정
        transform=transform,  # 미리 정의한 transform
        tokenizer=tokenizer ,  # 미리 정의한 토크나이저
        max_length=40
    )
    for split_name, split_file in split_files.items()
}

if __name__ == "__main__":
    
    # 디바이스 지정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DataLoader 생성
    train_loader = DataLoader(datasets['train'], batch_size=32, shuffle=False, num_workers=4)
    val_loader = DataLoader(datasets['validation'], batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(datasets['test'], batch_size=32, shuffle=False, num_workers=4)

    # CLIP 모델 생성
    model = CLIPModel()

    # Optimizer 생성
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 학습 진행
    num_epochs = 5  # 원하는 에폭 수
    best_val_loss = float('inf')

    # 손실 기록용 리스트
    train_loss_history = []
    val_loss_history = []

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        step = 0

        # tqdm progress bar로 학습 배치 진행 상황 표시
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch in pbar:
            step += 1
            # 배치 데이터를 디바이스로 이동
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            masks = batch["mask"].to(device)
            
            inputs = {
                "image": images,
                "input_ids": input_ids,
                "mask": masks
            }

            optimizer.zero_grad()

            loss = model(inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # 10 스텝마다 현재 Loss 출력
            if step % 1 == 0:
                pbar.set_postfix({"Loss": loss.item()})
                
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # 검증 루프
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                masks = batch["mask"].to(device)
                
                inputs = {
                    "image": images,
                    "input_ids": input_ids,
                    "mask": masks
                }

                loss = model(inputs)
                running_val_loss += loss.item() * images.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # 검증 손실이 개선되면 모델 저장
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_clip_model.pth")
            print("Best model saved!")
    
    # 테스트 루프: 저장된 모델을 로드해서 평가
    model.load_state_dict(torch.load("best_clip_model.pth"))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            masks = batch["mask"].to(device)
            
            inputs = {
                "image": images,
                "input_ids": input_ids,
                "mask": masks
            }

            loss = model(inputs)
            test_loss += loss.item() * images.size(0)
            
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    # 학습/검증 손실 변화를 그래프로 출력
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_loss_history, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 학습이 완료된 모델을 저장
    torch.save(model.state_dict(), "trained_CLIP.pth")
    print("학습이 완료되어 모델을 'trained_CLIP.pth'로 저장했습니다.")