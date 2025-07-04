import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from chromadb import PersistentClient, Settings
from typing import List, Dict, Tuple, Optional, Union, Any
import os

class FurnitureVectorSearch:
    """
    Класс для поиска похожей мебели по векторной базе данных на основе сегментированных изображений.
    
    Пример использования:
    ```python
    # Инициализация сегментатора и поисковика
    segmenter = FurnitureSegmenter(model_type="ade")
    searcher = FurnitureVectorSearch(db_path="./chroma_db", collection_name="furniture_clip")
    
    # Сегментация и поиск
    results = segmenter.segment_image("room.jpg")
    similar_items = searcher.find_similar_furniture(results)
    
    # Вывод результатов
    searcher.display_results(similar_items)
    ```
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        db_path: str = "./chroma_db",
        collection_name: str = "furniture_clip",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_similarity_threshold: float = 0.6
    ):
        """
        Инициализация векторного поисковика мебели.
        
        Args:
            clip_model_name: Название модели CLIP для извлечения эмбеддингов
            db_path: Путь к директории с векторной базой данных Chroma
            collection_name: Название коллекции в базе данных
            device: Устройство для вычислений ("cuda" или "cpu")
            min_similarity_threshold: Минимальный порог косинусной близости (0-1)
                                     Чем ближе к 1, тем больше похожесть требуется
        """
        self.device = device
        self.clip_model_name = clip_model_name
        self.db_path = db_path
        self.collection_name = collection_name
        self.min_similarity_threshold = min_similarity_threshold
        
        # Максимальное косинусное расстояние (для фильтрации результатов)
        # Косинусное расстояние = 1 - косинусная близость
        self.max_distance = 1 - self.min_similarity_threshold
        
        # Инициализация моделей и базы данных
        self._init_clip_model()
        self._init_vector_db()
    
    def _init_clip_model(self):
        """Инициализация модели CLIP для извлечения эмбеддингов."""
        print(f"Загрузка CLIP модели {self.clip_model_name}...")
        
        try:
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            print("CLIP модель успешно загружена")
        except Exception as e:
            print(f"Ошибка при загрузке CLIP модели: {e}")
            raise
    
    def _init_vector_db(self):
        """Инициализация векторной базы данных Chroma."""
        print(f"Подключение к базе данных Chroma в {self.db_path}...")
        
        try:
            self.chroma_client = PersistentClient(settings=Settings(persist_directory=self.db_path))
            self.collection = self.chroma_client.get_or_create_collection(self.collection_name)
            print(f"Успешное подключение к коллекции '{self.collection_name}'")
        except Exception as e:
            print(f"Ошибка при подключении к базе данных: {e}")
            raise
    
    def get_image_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Получение векторного представления изображения с помощью CLIP.
        
        Args:
            image: Изображение (путь к файлу, объект PIL.Image или numpy массив)
            
        Returns:
            np.ndarray: Векторное представление изображения
        """
        # Преобразование изображения в формат PIL.Image, если необходимо
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8')).convert("RGB")
        elif isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")
        
        # Получение векторного представления
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
        
        # Возвращаем плоский numpy массив
        return embedding.cpu().numpy().flatten()
    
    def find_similar_furniture(
        self, 
        segmentation_results: Dict,
        top_n: int = 4,
        custom_threshold: Optional[float] = None
    ) -> Dict[str, List[Dict]]:
        """
        Поиск похожей мебели для всех сегментированных объектов.
        
        Args:
            segmentation_results: Результаты сегментации от метода FurnitureSegmenter.segment_image()
            top_n: Количество похожих предметов для каждого объекта
            custom_threshold: Пользовательский порог близости (None для использования стандартного)
            
        Returns:
            Dict[str, List[Dict]]: Словарь с результатами поиска для каждого сегментированного объекта
        """
        # Получаем маски и информацию о классах из результатов сегментации
        masks = segmentation_results["masks"]
        class_names = segmentation_results["class_names"]
        orig_image = segmentation_results["orig_pil_image"]
        
        # Итоговые результаты
        search_results = {}
        
        # Порог для фильтрации
        threshold = custom_threshold if custom_threshold is not None else self.min_similarity_threshold
        # max_distance = 1 - threshold
        max_distance = 60
        
        # Обрабатываем каждый найденный объект
        for i, (mask, class_name) in enumerate(zip(masks, class_names)):
            print(f"Поиск похожей мебели для: {class_name} (объект {i+1}/{len(masks)})")
            
            # Извлекаем изображение объекта с помощью маски
            object_image = self._extract_object_with_mask(orig_image, mask)
            
            # Если объект успешно извлечен
            if object_image is not None:
                # Получаем эмбеддинг для объекта
                object_embedding = self.get_image_embedding(object_image)
                
                # Выполняем запрос к базе данных
                results = self.collection.query(
                    query_embeddings=[object_embedding.tolist()],
                    n_results=top_n * 2  # Запрашиваем больше, чтобы потом отфильтровать
                )
                
                # Обработка результатов поиска
                similar_items = []
                
                for j, (metadata, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
                    # Фильтрация по порогу косинусного расстояния
                    if distance <= max_distance:
                        similar_items.append({
                            "metadata": metadata,
                            "distance": distance,
                            "similarity": 1 - distance,  # Косинусная близость
                            "rank": j + 1
                        })
                
                # Сортировка по близости (наибольшая близость в начале)
                similar_items.sort(key=lambda x: x["similarity"], reverse=True)
                
                # Ограничение количества результатов до top_n
                similar_items = similar_items[:top_n]
                
                # Сохраняем результаты для текущего объекта
                key = f"{class_name}_{i}"
                search_results[key] = {
                    "class_name": class_name,
                    "object_index": i,
                    "similar_items": similar_items,
                    "object_image": object_image
                }
                
                print(f"  Найдено похожих предметов: {len(similar_items)}")
            else:
                print(f"  Не удалось извлечь изображение для объекта {class_name}_{i}")
        
        return search_results
    
    def _extract_object_with_mask(self, image: Image.Image, mask: np.ndarray) -> Optional[Image.Image]:
        """
        Извлечение объекта из изображения с использованием маски.
        
        Args:
            image: Исходное изображение
            mask: Маска объекта
            
        Returns:
            Optional[Image.Image]: Извлеченное изображение объекта или None, если извлечение не удалось
        """
        # Конвертируем изображение в numpy массив для обработки
        image_np = np.array(image)
        
        # Проверяем, что маска не пустая
        if not np.any(mask):
            return None
        
        # Создаем RGBA изображение с прозрачным фоном
        rgba = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
        
        # Копируем RGB каналы для пикселей объекта
        rgba[mask, :3] = image_np[mask]
        
        # Устанавливаем альфа канал (255 для объекта, 0 для фона)
        rgba[mask, 3] = 255
        
        # Преобразуем в PIL изображение
        pil_image = Image.fromarray(rgba)
        
        # Обрезаем до границ объекта
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Добавляем небольшой отступ
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(rgba.shape[1] - 1, x_max + padding)
            y_max = min(rgba.shape[0] - 1, y_max + padding)
            
            # Обрезаем изображение
            cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
            return cropped_image
        
        return None
    
    def display_results(
        self, 
        search_results: Dict[str, Dict],
        show_plot: bool = True,
        figsize: Tuple[int, int] = (15, 10)
    ) -> Optional[Dict[str, Any]]:
        """
        Визуализация результатов поиска.
        
        Args:
            search_results: Результаты поиска от метода find_similar_furniture
            show_plot: Отображать ли график
            figsize: Размер фигуры
            
        Returns:
            Optional[Dict[str, Any]]: Словарь с фигурами matplotlib, если show_plot=False
        """
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
        if not search_results:
            print("Нет результатов для отображения")
            return None
        
        figures = {}
        
        # Для каждого найденного объекта создаем отдельный график
        for obj_key, result in search_results.items():
            class_name = result["class_name"]
            object_index = result["object_index"]
            similar_items = result["similar_items"]
            object_image = result["object_image"]
            
            if not similar_items:
                print(f"Нет похожих предметов для {obj_key}")
                continue
            
            # Создаем фигуру
            fig = plt.figure(figsize=figsize)
            figures[obj_key] = fig
            
            # Количество результатов для отображения
            n_results = len(similar_items)
            
            # Размещение сетки графиков
            grid_size = n_results + 1
            cols = min(5, grid_size)
            rows = (grid_size + cols - 1) // cols
            
            # Отображаем исходный объект
            plt.subplot(rows, cols, 1)
            plt.imshow(object_image)
            plt.title(f"Исходный объект: {class_name}", fontsize=10)
            plt.axis('off')
            
            # Отображаем похожие предметы
            for i, item in enumerate(similar_items):
                metadata = item["metadata"]
                similarity = item["similarity"]
                
                # Получаем путь к изображению из метаданных
                image_path = metadata.get("image_path", "")
                
                plt.subplot(rows, cols, i + 2)
                
                if os.path.exists(image_path):
                    try:
                        img = Image.open(image_path)
                        plt.imshow(img)
                    except Exception:
                        plt.text(0.5, 0.5, "Ошибка загрузки", ha='center', va='center')
                else:
                    plt.text(0.5, 0.5, "Изображение\nотсутствует", ha='center', va='center')
                
                # Формируем заголовок с информацией о предмете
                title = f"Близость: {similarity:.2f}"
                if "product_name" in metadata:
                    title += f"\n{metadata['product_name']}"
                if "price" in metadata:
                    title += f"\nЦена: {metadata['price']}"
                
                plt.title(title, fontsize=9)
                plt.axis('off')
            
            plt.tight_layout()
            plt.suptitle(f"Поиск похожей мебели: {class_name} (объект {object_index})", fontsize=12)
            plt.subplots_adjust(top=0.9)
            
            if show_plot:
                plt.show()
        
        if not show_plot:
            return figures
        return None
    
    def add_images_to_database(
        self, 
        image_paths: List[str],
        metadatas: Optional[List[Dict]] = None,
        batch_size: int = 100
    ):
        """
        Добавление изображений в векторную базу данных.
        
        Args:
            image_paths: Список путей к изображениям
            metadatas: Список метаданных для каждого изображения
            batch_size: Размер пакета для добавления в базу данных
        """
        if not image_paths:
            print("Список изображений пуст")
            return
        
        if metadatas is None:
            # Если метаданные не предоставлены, создаем пустые словари
            metadatas = [{} for _ in range(len(image_paths))]
        
        # Проверяем, что количество метаданных соответствует количеству изображений
        if len(metadatas) != len(image_paths):
            raise ValueError("Количество метаданных должно соответствовать количеству изображений")
        
        # Дополняем метаданные путями к изображениям
        for i, path in enumerate(image_paths):
            metadatas[i]["image_path"] = path
        
        # Обработка изображений пакетами
        total_images = len(image_paths)
        print(f"Добавление {total_images} изображений в базу данных...")
        
        for i in range(0, total_images, batch_size):
            batch_end = min(i + batch_size, total_images)
            current_batch = image_paths[i:batch_end]
            current_metadatas = metadatas[i:batch_end]
            
            print(f"Обработка пакета {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}: "
                  f"изображения {i+1}-{batch_end} из {total_images}")
            
            # Получаем эмбеддинги для текущего пакета
            embeddings = []
            ids = []
            
            for j, path in enumerate(current_batch):
                try:
                    # Создаем уникальный идентификатор для изображения
                    image_id = f"img_{i+j}_{os.path.basename(path)}"
                    ids.append(image_id)
                    
                    # Получаем эмбеддинг
                    embedding = self.get_image_embedding(path)
                    embeddings.append(embedding.tolist())
                    
                except Exception as e:
                    print(f"Ошибка обработки изображения {path}: {e}")
            
            # Добавляем пакет в базу данных
            if embeddings:
                try:
                    self.collection.add(
                        embeddings=embeddings,
                        metadatas=current_metadatas[0:len(embeddings)],
                        ids=ids
                    )
                    print(f"Добавлено {len(embeddings)} изображений в базу данных")
                except Exception as e:
                    print(f"Ошибка при добавлении в базу данных: {e}")
        
        print(f"Всего добавлено изображений в базу данных: {self.collection.count()}")
    
    def update_database_from_directory(
        self, 
        directory_path: str,
        metadata_extractor: Optional[callable] = None,
        file_extensions: List[str] = ['.jpg', '.jpeg', '.png'],
        recursive: bool = True
    ):
        """
        Обновление базы данных изображениями из директории.
        
        Args:
            directory_path: Путь к директории с изображениями
            metadata_extractor: Функция для извлечения метаданных из имени файла или пути
            file_extensions: Список расширений файлов для обработки
            recursive: Обрабатывать ли поддиректории рекурсивно
        """
        image_paths = []
        metadatas = []
        
        # Функция по умолчанию для извлечения метаданных
        def default_metadata_extractor(path):
            return {"filename": os.path.basename(path)}
        
        # Используем предоставленную функцию или функцию по умолчанию
        extractor = metadata_extractor or default_metadata_extractor
        
        # Рекурсивно обходим директорию
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    path = os.path.join(root, file)
                    image_paths.append(path)
                    try:
                        metadata = extractor(path)
                        metadatas.append(metadata)
                    except Exception as e:
                        print(f"Ошибка извлечения метаданных для {path}: {e}")
                        metadatas.append({"filename": file})
            
            # Прерываем рекурсию, если recursive=False
            if not recursive:
                break
        
        # Добавляем найденные изображения в базу данных
        print(f"Найдено {len(image_paths)} изображений в директории {directory_path}")
        self.add_images_to_database(image_paths, metadatas)
    
    def search_by_image(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        top_n: int = 4,
        min_similarity: Optional[float] = None
    ) -> List[Dict]:
        """
        Поиск похожих изображений по изображению-запросу.
        
        Args:
            image: Изображение для поиска (путь, объект PIL.Image или numpy массив)
            top_n: Количество результатов
            min_similarity: Минимальная косинусная близость (None для использования стандартной)
            
        Returns:
            List[Dict]: Список найденных похожих изображений с метаданными
        """
        # Получаем эмбеддинг для изображения-запроса
        query_embedding = self.get_image_embedding(image)
        
        # Определяем порог
        threshold = min_similarity if min_similarity is not None else self.min_similarity_threshold
        max_distance = 1 - threshold
        
        # Выполняем запрос к базе данных
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_n * 2  # Запрашиваем больше для последующей фильтрации
        )
        
        # Обработка результатов поиска
        similar_items = []
        
        for i, (metadata, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
            # Фильтрация по порогу косинусного расстояния
            if distance <= max_distance:
                similar_items.append({
                    "metadata": metadata,
                    "distance": distance,
                    "similarity": 1 - distance,  # Косинусная близость
                    "rank": i + 1
                })
        
        # Сортировка по близости (наибольшая близость в начале)
        similar_items.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Ограничение количества результатов
        return similar_items[:top_n]