import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import DistilBertTokenizer
from CLIP import CLIPModel

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

class ZeroShotImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): 이미지들이 저장된 폴더 경로.
            transform (callable, optional): 이미지에 적용할 변환(transform) 함수.
        """
        self.image_dir = image_dir
        # 지정된 폴더 내 모든 이미지 파일 경로 읽기
        self.image_paths = glob.glob(os.path.join(image_dir, "*.*"))
        self.transform = transform
        self.labels = 'labels.txt'
        with open("labels.txt", "r") as f:
            self.labels = [int(line.strip()) for line in f if line.strip()]

        # 리스트를 PyTorch 텐서로 변환
        self.labels_tensor = torch.tensor(self.labels)

    def __len__(self):
        # 데이터셋에 포함된 이미지 수 반환
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 주어진 인덱스의 이미지 파일 경로 가져오기
        img_path = self.image_paths[idx]
        # 이미지를 RGB 모드로 열기
        image = Image.open(img_path).convert("RGB")
        # transform이 지정되어 있다면 적용
        if self.transform:
            image = self.transform(image)
        return image, self.labels_tensor[idx]

# 예제: Dataset과 DataLoader 사용하기
if __name__ == "__main__":
    # 이미지 전처리 파이프라인 (예: resize, center crop, tensor 변환, 정규화)
    transform = transforms.Compose([
        transforms.Resize(256),                # 짧은 변을 256으로 리사이즈
        transforms.CenterCrop(224),            # 224x224 크기로 중앙 자르기
        transforms.ToTensor(),                 # PIL 이미지를 Tensor로 변환 (값 범위: [0, 1])
        transforms.Normalize(                  # 정규화 (ImageNet 기준)
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 모델 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel(return_logits=True)
    # 저장된 모델 파라미터 로드 (strict=False 옵션으로 누락 키 무시)
    model.load_state_dict(torch.load("best_model.pth", map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    # 데이터셋 생성
    dataset = ZeroShotImageDataset("zeroshot_images", transform=transform)
    # DataLoader 생성 (배치 단위로 불러오기)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # "a photo of [CLS] 토크나이즈"
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokens = tokenizer(texts, padding=True, return_tensors="pt")

    with torch.no_grad():
        correct = 0.0
        # 텍스트 임베딩은 고정해 두고 사용
        text_embedding = model.text_encoder(tokens["input_ids"], tokens["attention_mask"])  # shape: (10, proj_dim)
        for image, labels in dataloader:
            image_embedding = model.image_encoder(image)  # shape: (batch, proj_dim)
            
            # logit 계산: 이미지 임베딩과 텍스트 임베딩의 내적에 temperature 스케일 적용
            logits = image_embedding @ text_embedding.T * torch.exp(model.temperature)
            # logits의 shape: (1, num_texts) -> squeeze해서 (num_texts,)로 변환
            logits = logits.squeeze(0)

            correct += (logits.argmax(dim=1) == labels).float().sum().item()
            
        print(f"Accuracy : {correct / 10000}")