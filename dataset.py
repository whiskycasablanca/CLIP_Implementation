import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
from transformers import DistilBertTokenizer  # 필요한 임포트 추가

class Flickr8kDataset(Dataset):
    def __init__(self, 
                 img_folder='images',
                 caption_file='captions.txt',
                 split_file=None,   # 추가: 분할 파일 경로 (예: Flickr_8k.testImages.txt)
                 transform=T.Compose([
                           T.Resize((224,224)),
                           T.ToTensor(),
                           # clip의 정규화 파라미터를 따름
                           T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
                           ]),
                 tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
                 max_length=40):    # 토큰의 90% 분위수=38이므로 40으로 설정함
    
        self.img_folder = img_folder
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1) 캡션 파일 읽어서 {이미지파일명: [캡션1, 캡션2, ...]} 형태로 저장
        self.img2captions = {}
        # 캡션 읽어서 f로 저장
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 각 라인의 형태: 1026685415_0431cbf574.jpg,A wet black dog is carrying a green toy through the grass
                # 쉼표 기준으로 분리하기
                img_cap, caption_text = line.split(',', 1)
                # 한 이미지에 5개의 캡션이 대응되므로 value를 리스트로 설정하기
                if img_cap not in self.img2captions:
                    self.img2captions[img_cap] = []
                # key-value 쌍 추가하기
                self.img2captions[img_cap].append(caption_text)

        # 2) 전체 이미지 파일 리스트 (중복 없이) 생성 ## 출력해서 어떻게 생겼는지 확인해볼 필요 있을듯
        self.image_files = list(self.img2captions.keys())
        """사용되는 이유: 
        인덱싱: __getitem__(self, idx) 구현 시, self.image_files[idx]와 같이 특정 인덱스의 이미지 파일명을 쉽게 가져올 수 있습니다.
        정렬 보장: 딕셔너리는 순서가 보장되지 않기 때문에, 리스트로 변환해두면 인덱스에 따라 일정한 순서로 데이터를 불러올 수 있습니다.
        DataLoader와의 호환: DataLoader는 len(dataset)과 __getitem__에 기반하여 데이터를 로드합니다. len(dataset)은 보통 self.image_files의 길이로 계산되므로, 이 부분이 있으면 전체 샘플 수를 바로 알 수 있습니다."""
        
        # 3) train/val/test 에 해당하는 이미지 파일만 사용하도록 필터링
        with open(split_file, 'r', encoding='utf-8') as f:
            split_images = set(line.strip() for line in f if line.strip())
        # 분할 파일에 포함된 이미지 파일만 남김
        self.image_files = [img for img in self.image_files if img in split_images]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # (1) 이미지 로드
        img_name = self.image_files[idx]  # 예) "1000268201_693b08cb0e.jpg"
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert('RGB')

        # (2) 랜덤 캡션 1개 선택
        captions = self.img2captions[img_name]  # 5개 중 하나를 뽑음
        caption = random.choice(captions)

        # (3) 이미지 전처리 => 논문에서 나온 것처럼 resizing만 하면 될듯. 확인해보니 이미지별로 사이즈가 제각각임
        if self.transform:
            image = self.transform(image)

        # (4) 텍스트 토큰화
        # DistilBertTokenizer 기준 (batch_first=True 형태)
        text_inputs = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # text_inputs 출력 예시: {'input_ids': tensor(...), 'attention_mask': tensor(...)}
        # 텐서 사이즈가 (1, sequence_length)로 설정되므로 불필요한 첫번째 차원 지우기
        for k, v in text_inputs.items():
            text_inputs[k] = v.squeeze(0)

        return {
         "image": image,
         "input_ids": text_inputs["input_ids"],
         "mask": text_inputs["attention_mask"]
        }