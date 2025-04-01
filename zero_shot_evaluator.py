import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

class ZeroShotEvaluator:
    def __init__(self):
        pass

    def split_dataset(self, embeddings_dict, test_size=0.2, random_state=42, min_samples_per_class=2):
        """
        Разделяет датасет на train и test наборы.
        Для классов с числом образцов ниже min_samples_per_class все данные идут в train.
        """
        train_data = {'embeddings': [], 'labels': [], 'paths': []}
        test_data = {'embeddings': [], 'labels': [], 'paths': []}

        random.seed(random_state)
        for class_name, samples in embeddings_dict.items():
            n_samples = len(samples)
            if n_samples < min_samples_per_class:
                print(f"Предупреждение: класс {class_name} содержит менее {min_samples_per_class} образцов ({n_samples}). Все данные идут в train.")
                for sample in samples:
                    train_data['embeddings'].append(sample['embedding'])
                    train_data['labels'].append(class_name)
                    train_data['paths'].append(sample['path'])
            else:
                random.shuffle(samples)
                split_idx = int(len(samples) * (1 - test_size))
                for sample in samples[:split_idx]:
                    train_data['embeddings'].append(sample['embedding'])
                    train_data['labels'].append(class_name)
                    train_data['paths'].append(sample['path'])
                for sample in samples[split_idx:]:
                    test_data['embeddings'].append(sample['embedding'])
                    test_data['labels'].append(class_name)
                    test_data['paths'].append(sample['path'])
        return {
            'train': (np.array(train_data['embeddings']), train_data['labels'], train_data['paths']),
            'test': (np.array(test_data['embeddings']), test_data['labels'], test_data['paths'])
        }
    
    def calculate_class_centroids(self, embeddings, labels):
        """Вычисляет центроиды для каждого класса"""
        class_embeddings = defaultdict(list)
        for emb, label in zip(embeddings, labels):
            class_embeddings[label].append(emb)
        centroids = {label: np.mean(embs, axis=0) for label, embs in class_embeddings.items()}
        return centroids

    def find_top_k_classes(self, query_embedding, centroids, k=3):
        """Находит top-k ближайших классов для заданного эмбеддинга"""
        distances = {}
        for label, centroid in centroids.items():
            distances[label] = np.linalg.norm(query_embedding - centroid)
        sorted_classes = sorted(distances.items(), key=lambda x: x[1])
        return [cls for cls, _ in sorted_classes[:k]]
    
    def evaluate_model(self, train_data, test_data, k=3):
        """
        Оценивает модель на тестовом наборе.
        Возвращает общую Top-k точность, подробные результаты и точность по классам.
        """
        X_train, y_train, _ = train_data
        X_test, y_test, paths_test = test_data

        centroids = self.calculate_class_centroids(X_train, y_train)
        results = []
        top_k_accuracy = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for idx in tqdm(range(len(X_test)), desc="Оценка модели"):
            query_embedding = X_test[idx]
            true_label = y_test[idx]
            image_path = paths_test[idx]

            top_k_classes = self.find_top_k_classes(query_embedding, centroids, k)
            is_correct = true_label in top_k_classes
            class_total[true_label] += 1
            if is_correct:
                class_correct[true_label] += 1
                top_k_accuracy += 1

            results.append({
                'image_path': image_path,
                'true_label': true_label,
                'predicted_classes': top_k_classes,
                'is_correct': is_correct
            })
        top_k_accuracy /= len(X_test)
        class_accuracies = {label: class_correct[label] / class_total[label] for label in class_total}
        return {
            'top_k_accuracy': top_k_accuracy,
            'detailed_results': results,
            'class_accuracies': class_accuracies
        }

    def print_evaluation_results(self, eval_results, k=3):
        """Выводит результаты оценки модели"""
        print(f"\nОбщая Top-{k} точность: {eval_results['top_k_accuracy']:.2%}")
        print("\nТочность по классам:")
        for label, acc in eval_results['class_accuracies'].items():
            print(f"{label}: {acc:.2%}")
        incorrect_predictions = [r for r in eval_results['detailed_results'] if not r['is_correct']]
        print(f"\nОбщее количество ошибок: {len(incorrect_predictions)}")
        for i, result in enumerate(incorrect_predictions):
            print(f"\nОшибка {i+1}:")
            print(f"Путь к изображению: {result['image_path']}")
            print(f"Правильный класс: {result['true_label']}")
            print(f"Предсказанный класс: {result['predicted_classes'][0]}")

    def visualize_results(self, eval_results, output_dir='results'):
        """Визуализирует результаты оценки: распределение предсказаний и точность по классам"""
        os.makedirs(output_dir, exist_ok=True)

        # График частоты появления классов в предсказаниях
        class_frequency = defaultdict(int)
        for result in eval_results['detailed_results']:
            for pred_class in result['predicted_classes']:
                class_frequency[pred_class] += 1
        sorted_freq = sorted(class_frequency.items(), key=lambda x: x[1], reverse=True)

        plt.figure(figsize=(15, 6))
        classes, frequencies = zip(*sorted_freq)
        plt.bar(range(len(classes)), frequencies)
        plt.xticks(range(len(classes)), classes, rotation=90)
        plt.title('Частота появления классов в top-1 предсказаниях')
        plt.xlabel('Класс')
        plt.ylabel('Частота')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_frequency.png'))
        plt.close()

        # График точности по классам
        plt.figure(figsize=(15, 6))
        classes = list(eval_results['class_accuracies'].keys())
        accuracies = list(eval_results['class_accuracies'].values())
        plt.bar(range(len(classes)), accuracies)
        plt.xticks(range(len(classes)), classes, rotation=90)
        plt.title('Точность по классам')
        plt.xlabel('Класс')
        plt.ylabel('Точность')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_accuracies.png'))
        plt.close()

def load_embeddings(embeddings_file):
    """Загружает эмбеддинги из файла (pickle)"""
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Оценка модели Zero-Shot")
    parser.add_argument('--embeddings_file', type=str, required=True, help="Путь к файлу с эмбеддингами (pickle)")
    parser.add_argument('--test_size', type=float, default=0.2, help="Доля тестовых данных")
    parser.add_argument('--top_k', type=int, default=3, help="Параметр top-k для оценки")
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings_file)
    evaluator = ZeroShotEvaluator()
    data = evaluator.split_dataset(embeddings, test_size=args.test_size)
    eval_results = evaluator.evaluate_model(data['train'], data['test'], k=args.top_k)
    evaluator.print_evaluation_results(eval_results, k=args.top_k)
    evaluator.visualize_results(eval_results)
    print("Анализ завершен!")
