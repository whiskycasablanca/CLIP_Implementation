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
from text_encoder import TextEncoder

#device설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 설정
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
texts = ["This is a sample sentence.", "This is another example."]
#토크나이저로 텍스트 인코더의 인풋 생성
inputs= tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device) 

