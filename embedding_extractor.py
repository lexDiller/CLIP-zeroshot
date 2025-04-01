import os
import torch
import open_clip
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle

class EmbeddingExtractor:
    def __init__(self, model_name='ViT-L/14', pretrained='laion2b_s32b_b82k'):
        # Инициализация модели и преобразований
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Используется устройство: {self.device}")
        self.model = self.model.to(self.device)
    
    def get_image_embedding(self, image_path):
        """Извлекает эмбеддинг для одного изображения"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0]
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {str(e)}")
            return None
    
    def process_dataset(self, data_path):
        """
        Обрабатывает датасет, извлекая эмбеддинги для изображений.
        Результат — словарь, где ключи – классы, а значения — список словарей с эмбеддингом и путем к изображению.
        """
        embeddings = defaultdict(list)
        print("Сбор данных...")
        for class_folder in tqdm(os.listdir(data_path), desc="Обработка классов"):
            class_path = os.path.join(data_path, class_folder)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(class_path, image_file)
                        embedding = self.get_image_embedding(image_path)
                        if embedding is not None:
                            embeddings[class_folder].append({'embedding': embedding, 'path': image_path})
        return embeddings

    def save_embeddings(self, embeddings, output_file):
        """Сохраняет извлечённые эмбеддинги в файл (pickle)"""
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Эмбеддинги сохранены в {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Извлечение эмбеддингов из датасета изображений")
    parser.add_argument('--data_path', type=str, required=True, help="Путь к папке с данными")
    parser.add_argument('--output_file', type=str, required=True, help="Файл для сохранения эмбеддингов (например, embeddings.pkl)")
    args = parser.parse_args()

    extractor = EmbeddingExtractor()
    embeddings = extractor.process_dataset(args.data_path)
    extractor.save_embeddings(embeddings, args.output_file)
