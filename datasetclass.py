class FlickrDataset(Dataset):
    def __init__(self, data_dir, max_len, img_size):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_len = max_len

        self.images_dir = self.data_dir/"Images"
        self.img_paths = sorted(list(map(str, self.images_dir.glob("**/*.jpg"))))

        self.transformer = get_train_transformer(img_size=img_size)

        self.captions = defaultdict(list)
        with open(self.data_dir/"captions.txt", mode="r") as f:
            for line in f:
                line = line.strip()
                if ".jpg" in line:
                    split_idx = re.search(pattern=r"(.jpg)", string=line).span()[1]
                    img_path = str(self.images_dir/line[: split_idx])
                    text = line[split_idx + 1:].replace(" .", ".")
                    if img_path in self.img_paths:
                        self.captions[img_path].append(text)
def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        image = self.transformer(image)

        texts = self.captions[img_path]
        text = random.choice(texts)
        token_ids = encode(text, tokenizer=self.tokenizer, max_len=self.max_len)
        return image, token_ids