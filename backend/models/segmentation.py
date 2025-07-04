import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import cv2
import os
import random
from typing import List, Dict, Tuple, Optional, Union, Literal

class FurnitureSegmenter:
    """
    Класс для сегментации мебели на изображениях с использованием моделей Mask2Former.
    
    Пример использования:
    ```python
    segmenter = FurnitureSegmenter(model_type="ade")
    results = segmenter.segment_image("path/to/image.jpg")
    segmenter.visualize_results(results)
    segmenter.save_segmented_objects(results, output_dir="./segmented_furniture")
    ```
    """
    
    # Словари классов мебели для различных моделей
    COCO_FURNITURE_CLASSES = {
        # Основная мебель
        56: "chair",          # стул
        57: "couch",          # диван
        59: "bed",            # кровать
        60: "dining table",   # обеденный стол
        61: "toilet",         # унитаз
        62: "tv",             # телевизор
        68: "microwave",      # микроволновка
        69: "oven",           # духовка
        70: "toaster",        # тостер
        72: "refrigerator",   # холодильник
        74: "clock",          # часы
        75: "vase",           # ваза
        85: "curtain",        # штора 
        95: "pillow",         # подушка 
        92: "light",          # лампа/источник света 
        132: "rug",           # ковер 
        93: "mirror",         # зеркало 
        104: "shelf",         # полка 
        84: "counter",        # стойка 
    }
    
    ADE_FURNITURE_CLASSES = {
        # Основная мебель
        7: "bed",               # кровать
        10: "cabinet",          # шкаф (кухонный/офисный)
        15: "table",            # стол (общий)
        19: "chair",            # стул
        23: "sofa",             # диван
        30: "armchair",         # кресло
        33: "desk",             # письменный стол
        35: "wardrobe",         # гардероб
        44: "chest of drawers", # комод
        64: "coffee table",     # журнальный столик
        110: "stool",           # табурет
        
        # Сантехника и кухня
        37: "bathtub",          # ванна
        47: "sink",             # раковина
        50: "refrigerator",     # холодильник
        65: "toilet",           # унитаз
        71: "stove",            # плита
        118: "oven",            # духовка
        124: "microwave",       # микроволновка
        
        # Декор и текстиль
        18: "curtain",          # штора
        22: "painting",         # картина
        27: "mirror",           # зеркало
        28: "rug",              # ковер
        39: "cushion",          # подушка (декоративная)
        57: "pillow",           # подушка (спальная)
        85: "chandelier",       # люстра
        97: "ottoman",          # пуфик
        135: "vase",            # ваза
        
        # Хранение и полки
        24: "shelf",            # полка
        62: "bookcase",         # книжный шкаф
        
        # Освещение
        36: "lamp",             # лампа (настольная/торшер)
        82: "light",            # источник света
        134: "sconce",          # настенный светильник
        
        # Дополнительные элементы
        49: "fireplace",        # камин
        70: "countertop",       # столешница
        73: "kitchen island",   # кухонный остров
        99: "buffet",           # сервант
        146: "radiator",        # радиатор отопления
        148: "clock",           # часы
    }
    
    # Модели и их конфигурации
    MODEL_CONFIGS = {
        "coco": {
            "processor": "facebook/mask2former-swin-large-coco-panoptic",
            "model": "facebook/mask2former-swin-large-coco-panoptic",
            "classes": COCO_FURNITURE_CLASSES
        },
        "ade": {
            "processor": "facebook/mask2former-swin-large-ade-panoptic",
            "model": "facebook/mask2former-swin-large-ade-panoptic",
            "classes": ADE_FURNITURE_CLASSES
        }
    }
    
    def __init__(
        self, 
        model_type: Literal["coco", "ade"] = "ade",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        custom_model_path: Optional[str] = None,
        custom_processor_path: Optional[str] = None,
        custom_classes: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Инициализация сегментатора мебели.
        
        Args:
            model_type: Тип модели ("coco" или "ade")
            device: Устройство для вычислений ("cuda" или "cpu")
            custom_model_path: Путь к пользовательской модели (если есть)
            custom_processor_path: Путь к пользовательскому процессору (если есть)
            custom_classes: Пользовательский словарь классов (если есть)
            confidence_threshold: Порог уверенности для сегментации
        """
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.confidence_threshold = confidence_threshold
        
        # Если указаны пользовательские пути к модели и процессору
        if custom_model_path and custom_processor_path:
            self.processor_path = custom_processor_path
            self.model_path = custom_model_path
            self.furniture_classes = custom_classes or self.ADE_FURNITURE_CLASSES
        else:
            # Используем предопределенные конфигурации
            if model_type not in self.MODEL_CONFIGS:
                raise ValueError(f"Неподдерживаемый тип модели: {model_type}. Доступные типы: {list(self.MODEL_CONFIGS.keys())}")
            
            config = self.MODEL_CONFIGS[model_type]
            self.processor_path = config["processor"]
            self.model_path = config["model"]
            self.furniture_classes = config["classes"]
        
        self._load_models()
    
    def _load_models(self):
        """Загрузка моделей и процессоров для сегментации."""
        print(f"Загрузка моделей из {self.model_path}...")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.processor_path)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.model_path, 
                # torch_dtype=self.torch_dtype
            )
            
            # Перемещение модели на указанное устройство
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            print("Модели успешно загружены")
        except Exception as e:
            print(f"Ошибка при загрузке моделей: {e}")
            raise
    
    def _load_image(self, image_source: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """
        Загрузка изображения из различных источников.
        
        Args:
            image_source: Исходник изображения (путь, URL, объект PIL.Image или numpy массив)
            
        Returns:
            Image.Image: Загруженное изображение
        """
        if isinstance(image_source, str):
            # Загрузка из URL или файла
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_source)
        elif isinstance(image_source, np.ndarray):
            # Преобразование из numpy массива
            image = Image.fromarray(image_source.astype('uint8'))
        elif isinstance(image_source, Image.Image):
            # Уже объект Image
            image = image_source
        else:
            raise TypeError(f"Неподдерживаемый тип источника изображения: {type(image_source)}")
        
        # Убедимся, что изображение в формате RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    
    def _generate_random_colors(self, n: int) -> List[Tuple[float, float, float]]:
        """
        Генерация случайных цветов для визуализации сегментированных объектов.
        
        Args:
            n: Количество цветов для генерации
            
        Returns:
            List[Tuple[float, float, float]]: Список RGB цветов
        """
        colors = []
        for _ in range(n):
            color = (random.random(), random.random(), random.random())
            colors.append(color)
        return colors
    
    def segment_image(
        self, 
        image_source: Union[str, Image.Image, np.ndarray],
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Сегментация мебели на изображении.
        
        Args:
            image_source: Источник изображения (путь, URL, PIL.Image или numpy массив)
            threshold: Порог уверенности (если None, используется стандартный)
            
        Returns:
            Dict: Результаты сегментации со следующими ключами:
                - "image": исходное изображение (numpy массив)
                - "segments_info": информация о сегментированных объектах
                - "class_ids": ID классов для каждого объекта
                - "class_names": названия классов для каждого объекта
                - "masks": маски для каждого объекта
                - "orig_pil_image": исходное PIL изображение
        """
        # Установка порога уверенности
        conf_threshold = threshold if threshold is not None else self.confidence_threshold
        
        # Загрузка изображения
        image = self._load_image(image_source)
        
        # Сохраняем оригинальное PIL изображение
        orig_pil_image = image.copy()
        
        # Преобразование изображения для модели
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Переносим тензоры на указанное устройство
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Получение предсказаний
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Постобработка результатов
        result = self.processor.post_process_panoptic_segmentation(
            outputs, 
            target_sizes=[image.size[::-1]],
            threshold=conf_threshold,
        )[0]
        
        # Извлечение нужной информации
        segmentation = result["segmentation"]
        segments_info = result["segments_info"]
        
        # Фильтрация только объектов мебели
        furniture_segments = []
        furniture_masks = []
        furniture_class_ids = []
        furniture_class_names = []
        
        for segment in segments_info:
            class_id = segment["label_id"]
            # Проверяем, является ли объект мебелью
            if class_id in self.furniture_classes:
                furniture_segments.append(segment)
                furniture_class_ids.append(class_id)
                furniture_class_names.append(self.furniture_classes[class_id])
                
                # Создаем маску для этого объекта
                mask = (segmentation == segment["id"]).cpu().numpy()
                furniture_masks.append(mask)
        
        # Конвертирование изображения в массив numpy для обработки
        image_np = np.array(image)
        
        # Формирование результатов
        results = {
            "image": image_np,
            "segments_info": furniture_segments,
            "class_ids": furniture_class_ids,
            "class_names": furniture_class_names,
            "masks": furniture_masks,
            "orig_pil_image": orig_pil_image
        }
        
        return results
    
    def visualize_results(
        self, 
        results: Dict,
        show_individual_objects: bool = True,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (15, 10)
    ) -> Tuple[plt.Figure, ...]:
        """
        Визуализация результатов сегментации.
        
        Args:
            results: Результаты сегментации от метода segment_image
            show_individual_objects: Показывать ли отдельные объекты
            show_plot: Отображать ли графики сразу
            figsize: Размер фигуры для основного графика
            
        Returns:
            Tuple[plt.Figure, ...]: Фигуры matplotlib
        """
        # Извлечение данных из результатов
        image = results["image"]
        segments = results["segments_info"]
        class_ids = results["class_ids"]
        class_names = results["class_names"]
        masks = results["masks"]
        
        # Генерация случайных цветов для каждого класса
        colors = self._generate_random_colors(len(class_ids))
        
        # Создание основной фигуры
        main_fig = plt.figure(figsize=figsize)
        
        # Отображение исходного изображения
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Исходное изображение")
        plt.axis('off')
        
        # Создание и отображение изображения с масками
        masked_image = image.copy()
        for i, mask in enumerate(masks):
            # Применяем цветную маску с прозрачностью
            color_mask = np.zeros_like(image)
            color = [int(c * 255) for c in colors[i]]
            for c in range(3):
                color_mask[:, :, c] = color[c]
            
            # Применяем маску с легкой прозрачностью
            alpha = 0.4  # Уровень прозрачности
            mask_area = mask.astype(bool)
            masked_image[mask_area] = masked_image[mask_area] * (1 - alpha) + color_mask[mask_area] * alpha
        
        # Отображение изображения с масками
        plt.subplot(1, 2, 2)
        plt.imshow(masked_image)
        plt.title("Сегментированная мебель")
        plt.axis('off')
        
        # Добавляем аннотации с названиями классов на сегментированное изображение
        for i, (mask, class_name) in enumerate(zip(masks, class_names)):
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min = np.min(y_indices)
                x_center = np.mean(x_indices).astype(int)
                
                # Добавляем текстовую аннотацию с названием класса
                plt.annotate(
                    class_name,
                    xy=(x_center, y_min),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha='center',
                    color='white',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc=colors[i], alpha=0.7)
                )
        
        # Создание фигуры для отдельных объектов, если требуется
        objects_fig = None
        if show_individual_objects and len(class_ids) > 0:
            # Определяем количество строк и столбцов для подграфиков
            cols = min(3, len(class_ids))
            rows = (len(class_ids) + cols - 1) // cols
            
            objects_fig = plt.figure(figsize=(15, 5 * rows))
            
            for i, (mask, class_name) in enumerate(zip(masks, class_names)):
                plt.subplot(rows, cols, i + 1)
                
                # Вырезаем объект с помощью маски
                object_image = image.copy()
                # Создаем черный фон
                bg = np.zeros_like(object_image)
                # Копируем объект на черный фон
                bg[mask] = object_image[mask]
                
                # Обрезаем изображение до размеров объекта
                # Находим границы маски
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    
                    # Добавляем небольшой отступ
                    padding = 10
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(bg.shape[1] - 1, x_max + padding)
                    y_max = min(bg.shape[0] - 1, y_max + padding)
                    
                    # Обрезаем изображение
                    cropped_object = bg[y_min:y_max, x_min:x_max]
                    
                    plt.imshow(cropped_object)
                    plt.title(f"Класс: {class_name}")
                    plt.axis('off')
                else:
                    plt.text(0.5, 0.5, "Пустая маска", horizontalalignment='center')
        
        # Отображение фигур
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        # Возвращаем фигуры для дальнейшего использования
        if objects_fig:
            return main_fig, objects_fig
        else:
            return (main_fig,)
    
    def save_segmented_objects(
        self, 
        results: Dict,
        output_dir: str = "./segmented_furniture",
        save_masks: bool = True,
        save_individual_objects: bool = True,
        save_visualization: bool = True,
        file_prefix: str = "",
    ) -> Dict[str, List[str]]:
        """
        Сохранение сегментированных объектов и масок.
        
        Args:
            results: Результаты сегментации от метода segment_image
            output_dir: Директория для сохранения результатов
            save_masks: Сохранять ли маски объектов
            save_individual_objects: Сохранять ли отдельные объекты
            save_visualization: Сохранять ли визуализацию с масками
            file_prefix: Префикс для имени файлов
            
        Returns:
            Dict[str, List[str]]: Пути к сохраненным файлам по категориям
        """
        # Извлечение данных из результатов
        image = results["image"]
        masks = results["masks"]
        class_names = results["class_names"]
        
        # Создаем директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Для хранения путей к сохраненным файлам
        saved_files = {
            "objects": [],
            "masks": [],
            "visualizations": []
        }
        
        # Формирование префикса с временной меткой, если не указан
        if not file_prefix:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_prefix = f"furniture_{timestamp}_"
        
        # Сохранение отдельных объектов, если требуется
        if save_individual_objects:
            for i, (mask, class_name) in enumerate(zip(masks, class_names)):
                # Вырезаем объект с помощью маски
                object_image = image.copy()
                # Создаем прозрачный фон (RGBA)
                bg = np.zeros((object_image.shape[0], object_image.shape[1], 4), dtype=np.uint8)
                # Копируем RGB каналы объекта
                bg[mask, :3] = object_image[mask]
                # Устанавливаем альфа канал (прозрачность)
                bg[mask, 3] = 255  # Непрозрачно для объекта
                
                # Обрезаем изображение до размеров объекта
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    
                    # Добавляем небольшой отступ
                    padding = 10
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(bg.shape[1] - 1, x_max + padding)
                    y_max = min(bg.shape[0] - 1, y_max + padding)
                    
                    # Обрезаем изображение
                    cropped_object = bg[y_min:y_max, x_min:x_max]
                    
                    # Сохраняем изображение с прозрачным фоном
                    safe_class_name = class_name.replace(" ", "_")
                    filename = f"{output_dir}/{file_prefix}{safe_class_name}_{i}.png"
                    Image.fromarray(cropped_object).save(filename)
                    saved_files["objects"].append(filename)
                    print(f"Сохранено: {filename}")
                else:
                    print(f"Пропуск пустой маски для {class_name}_{i}")
        
        # Сохранение масок, если требуется
        if save_masks:
            for i, (mask, class_name) in enumerate(zip(masks, class_names)):
                mask_image = mask.astype(np.uint8) * 255
                safe_class_name = class_name.replace(" ", "_")
                mask_path = f"{output_dir}/{file_prefix}{safe_class_name}_{i}_mask.png"
                Image.fromarray(mask_image).save(mask_path)
                saved_files["masks"].append(mask_path)
                print(f"Сохранена маска: {mask_path}")
        
        # Сохранение визуализации с масками, если требуется
        if save_visualization:
            # Генерация случайных цветов
            colors = self._generate_random_colors(len(masks))
            
            # Создание визуализации
            masked_image = image.copy()
            for i, mask in enumerate(masks):
                color = [int(c * 255) for c in colors[i]]
                alpha = 0.4
                mask_area = mask.astype(bool)
                for c in range(3):
                    masked_image[mask_area, c] = masked_image[mask_area, c] * (1 - alpha) + color[c] * alpha
            
            # Добавляем аннотации с названиями классов
            for i, (mask, class_name) in enumerate(zip(masks, class_names)):
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    y_min = np.min(y_indices)
                    x_center = np.mean(x_indices).astype(int)
                    
                    # Добавляем текст с названием класса
                    cv2.putText(masked_image, class_name, (x_center, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Сохраняем полное сегментированное изображение
            full_segmented_path = f"{output_dir}/{file_prefix}full_segmentation.jpg"
            Image.fromarray(masked_image).save(full_segmented_path)
            saved_files["visualizations"].append(full_segmented_path)
            print(f"Сохранено сегментированное изображение: {full_segmented_path}")
        
        return saved_files
    
    def get_furniture_stats(self, results: Dict) -> Dict[str, Dict]:
        """
        Получение статистики по найденным объектам мебели.
        
        Args:
            results: Результаты сегментации от метода segment_image
            
        Returns:
            Dict[str, Dict]: Статистика по каждому объекту и общая статистика
        """
        masks = results["masks"]
        class_names = results["class_names"]
        
        # Словарь для хранения статистики
        stats = {
            "total_objects": len(class_names),
            "objects": [],
            "class_stats": {}
        }
        
        # Расчет статистики для каждого объекта
        total_area = 0
        for i, (mask, class_name) in enumerate(zip(masks, class_names)):
            # Площадь маски в пикселях
            area = mask.sum()
            total_area += area
            
            # Сохранение статистики
            object_stats = {
                "class_name": class_name,
                "area_pixels": int(area),
                "area_percentage": 0  # Будет обновлено позже
            }
            stats["objects"].append(object_stats)
            
            # Агрегация статистики по классам
            if class_name not in stats["class_stats"]:
                stats["class_stats"][class_name] = {
                    "count": 1,
                    "total_area": int(area)
                }
            else:
                stats["class_stats"][class_name]["count"] += 1
                stats["class_stats"][class_name]["total_area"] += int(area)
        
        # Обновление процентного соотношения площадей
        if total_area > 0:
            for obj in stats["objects"]:
                obj["area_percentage"] = round(obj["area_pixels"] / total_area * 100, 2)
            
            for class_name in stats["class_stats"]:
                stats["class_stats"][class_name]["area_percentage"] = round(
                    stats["class_stats"][class_name]["total_area"] / total_area * 100, 2
                )
        
        return stats
    
    def print_stats(self, stats: Dict):
        """
        Вывод статистики по найденным объектам мебели.
        
        Args:
            stats: Статистика от метода get_furniture_stats
        """
        print(f"Найдено предметов мебели: {stats['total_objects']}")
        
        if stats['total_objects'] > 0:
            print("\nСтатистика по объектам:")
            for i, obj in enumerate(stats["objects"]):
                print(f"{i+1}. {obj['class_name']}: {obj['area_pixels']} пикселей ({obj['area_percentage']}%)")
            
            print("\nСтатистика по классам:")
            for class_name, class_stats in stats["class_stats"].items():
                print(f"- {class_name}: {class_stats['count']} шт., {class_stats['total_area']} пикселей ({class_stats['area_percentage']}%)")