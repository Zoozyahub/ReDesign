import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional, Union, Tuple, Dict, Any
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    DDIMScheduler,
    AutoencoderKL
)


class InteriorGenerator:
    """
    Класс для генерации изображений интерьера с использованием Stable Diffusion и ControlNet.
    
    Пример использования:
    ```python
    generator = InteriorGenerator()
    result_path = generator.generate(
        input_image_path="room_sketch.png",
        user_prompt="Modern Style White bed with big cabinet",
        output_dir="./generated_interiors"
    )
    ```
    """
    
    def __init__(
        self,
        controlnet_model: str = "lllyasviel/control_v11p_sd15_canny",
        base_model: str = "SG161222/Realistic_Vision_V5.1_noVAE",
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        default_base_prompt: str = "interior design, 4k, detailed texture, professional lighting, interior visualization, architectural photography, no people, minimalist style",
        default_negative_prompt: str = "cartoon, anime, sketch, low-quality, blurry, deformed, distorted furniture, unrealistic proportions, bad shadows, poor lighting, oversaturated colors",
        canny_low_threshold: int = 100,
        canny_high_threshold: int = 200,
    ):
        """
        Инициализация генератора интерьера.
        
        Args:
            controlnet_model: Модель ControlNet для управления генерацией.
            base_model: Базовая модель Stable Diffusion.
            vae_model: Модель VAE для улучшенного декодирования изображений.
            device: Устройство для вычислений ('cuda' или 'cpu').
            default_base_prompt: Базовый промпт, добавляемый к пользовательскому промпту.
            default_negative_prompt: Негативный промпт для исключения нежелательных элементов.
            canny_low_threshold: Нижний порог для детектора Canny.
            canny_high_threshold: Верхний порог для детектора Canny.
        """
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.default_base_prompt = default_base_prompt
        self.default_negative_prompt = default_negative_prompt
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        
        # Загрузка моделей
        self._load_models(controlnet_model, base_model, vae_model)
    
    def _load_models(self, controlnet_model: str, base_model: str, vae_model: str):
        """
        Загрузка всех необходимых моделей.
        
        Args:
            controlnet_model: Путь к модели ControlNet.
            base_model: Путь к базовой модели SD.
            vae_model: Путь к модели VAE.
        """
        print("Загрузка моделей...")
        
        # Загрузка VAE
        vae = AutoencoderKL.from_pretrained(
            vae_model, 
            torch_dtype=self.torch_dtype
        )
        
        # Загрузка ControlNet
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model, 
            torch_dtype=self.torch_dtype
        )
        
        # Загрузка основной модели
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model, 
            controlnet=controlnet,
            vae=vae,
            safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        
        # Установка DDIM планировщика
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Оптимизация памяти
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        
        print("Модели успешно загружены.")
    
    def _create_canny_map(self, image: Image.Image) -> Image.Image:
        """
        Создание карты границ Canny для ControlNet.
        
        Args:
            image: Входное изображение PIL.
            
        Returns:
            Image.Image: Карта границ Canny.
        """
        image_np = np.array(image)
        canny = cv2.Canny(image_np, self.canny_low_threshold, self.canny_high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        return Image.fromarray(canny)
    
    def _prepare_image(self, image_path: str, max_dim: int = 768) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
        """
        Загрузка и подготовка изображения для генерации.
        
        Args:
            image_path: Путь к исходному изображению.
            max_dim: Максимальный размер по наибольшей стороне.
            
        Returns:
            Tuple[Image.Image, Image.Image, Tuple[int, int]]: 
                - Подготовленное входное изображение
                - Карта границ Canny
                - Оригинальные размеры изображения
        """
        # Загрузка изображения
        input_image = Image.open(image_path).convert("RGB")
        
        # Сохранение оригинальных размеров
        original_size = input_image.size
        
        # Изменение размера с сохранением пропорций
        width, height = input_image.size
        scale = min(max_dim / width, max_dim / height)
        new_width, new_height = int(width * scale), int(height * scale)
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Создание карты Canny
        canny_image = self._create_canny_map(input_image)
        
        return input_image, canny_image, original_size
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Постобработка для улучшения контраста и цветов.
        
        Args:
            image: Сгенерированное изображение.
            
        Returns:
            Image.Image: Улучшенное изображение.
        """
        # Конвертация в numpy для обработки с OpenCV
        img_np = np.array(image)
        
        # Улучшение контраста и яркости
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced_img)
    
    def generate(
        self,
        input_image_path: str,
        user_prompt: str,
        output_dir: str = "./generated",
        output_filename: str = None,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        apply_enhancement: bool = True,
        additional_prompt: str = "",
        negative_prompt: str = None,
        return_image: bool = True,
        kwargs: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Генерация изображения интерьера на основе входного изображения и промпта.
        
        Args:
            input_image_path: Путь к входному изображению.
            user_prompt: Пользовательский промпт для генерации.
            output_dir: Директория для сохранения результатов.
            output_filename: Имя выходного файла (если None, будет сгенерировано).
            width: Ширина выходного изображения.
            height: Высота выходного изображения.
            num_inference_steps: Количество шагов инференса.
            guidance_scale: Сила следования промпту (CFG scale).
            seed: Seed для воспроизводимости результатов (если None, случайный).
            apply_enhancement: Применять ли улучшение изображения.
            additional_prompt: Дополнительный пользовательский промпт.
            negative_prompt: Пользовательский негативный промпт (если None, используется стандартный).
            return_image: Возвращать ли изображение в памяти.
            kwargs: Дополнительные параметры для пайплайна генерации.
            
        Returns:
            Dict[str, Any]: Словарь с результатами:
                - "image": PIL изображение (если return_image=True)
                - "enhanced_image": Улучшенное PIL изображение (если apply_enhancement=True и return_image=True)
                - "original_path": Путь к сохраненному оригинальному изображению
                - "enhanced_path": Путь к сохраненному улучшенному изображению (если apply_enhancement=True)
        """
        # Проверка существования и создание директории
        os.makedirs(output_dir, exist_ok=True)
        
        # Генерация имени файла, если не указано
        if output_filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"interior_{timestamp}"
        
        # Подготовка изображения и создание карты Canny
        _, canny_image, original_size = self._prepare_image(input_image_path)
        
        # Формирование полного промпта
        full_prompt = f"{user_prompt}, {self.default_base_prompt}"
        if additional_prompt:
            full_prompt = f"{user_prompt}, {additional_prompt}, {self.default_base_prompt}"
        
        # Использование пользовательского или стандартного негативного промпта
        neg_prompt = negative_prompt if negative_prompt is not None else self.default_negative_prompt
        
        # Установка генератора случайных чисел для воспроизводимости
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Объединение стандартных и пользовательских дополнительных параметров
        generation_kwargs = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
        }
        if kwargs:
            generation_kwargs.update(kwargs)
        
        # Генерация изображения
        print(f"Генерация с промптом: '{full_prompt}'")
        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            image=canny_image,
            **generation_kwargs
        ).images[0]
        
        # Масштабирование результата обратно к оригинальному размеру
        result = result.resize(original_size, Image.LANCZOS)
        
        # Пути для сохранения файлов
        original_path = os.path.join(output_dir, f"{output_filename}.png")
        enhanced_path = os.path.join(output_dir, f"{output_filename}_enhanced.png")
        
        # Сохранение оригинального результата
        result.save(original_path)
        print(f"Сохранено изображение: {original_path}")
        
        # Результаты для возврата
        results = {
            "original_path": original_path,
        }
        
        if return_image:
            results["image"] = result
        
        # Применение улучшения, если требуется
        if apply_enhancement:
            enhanced_result = self._enhance_image(result)
            enhanced_result.save(enhanced_path)
            print(f"Сохранено улучшенное изображение: {enhanced_path}")
            results["enhanced_path"] = enhanced_path
            
            if return_image:
                results["enhanced_image"] = enhanced_result
        
        return results