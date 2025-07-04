import React, { useState, useRef, useEffect } from "react";
import "./generation.css";

export default function InteriorDesignGenerator() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isQuickMode, setIsQuickMode] = useState(true);
  const [roomType, setRoomType] = useState("");
  const [interiorStyle, setInteriorStyle] = useState("");
  const [textPrompt, setTextPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isGenerated, setIsGenerated] = useState(false);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isSliderDragging, setIsSliderDragging] = useState(false);
  const [imageAspectRatio, setImageAspectRatio] = useState(1.5); // Default aspect ratio (width/height)
  const [generatedImageUrl, setGeneratedImageUrl] = useState(null);
  const [recommendedFurniture, setRecommendedFurniture] = useState([]);
  const [isPublic, setIsPublic] = useState(false);

  const handleCheckboxChange = (event) => {
    setIsPublic(event.target.checked);
  };
  
  const fileInputRef = useRef(null);
  // const sliderRef = useRef(null);
  const imageContainerRef = useRef(null);
  
  const roomTypes = [
    "Гостиная", "Спальня", "Кухня", "Ванная комната", 
    "Столовая", "Домашний офис", "Детская комната", "Прихожая"
  ];

  const interiorStyles = [
      "Минимализм", "Скандинавский", "Индустриальный", "Современный", 
      "Классический", "Ар-деко", "Лофт", "Хай-тек", "Рустик"
  ];
  
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileDrop(file);
    }
  };
  
  const handleFileDrop = (file) => {
    if (file && file.type.match('image.*')) {
      setSelectedFile(file);
      
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.onload = () => {
          // Calculate and store the aspect ratio (width / height)
          setImageAspectRatio(img.width / img.height);
        };
        img.src = reader.result;
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
      
      // Reset generation state when new file is uploaded
      setIsGenerated(false);
      setGeneratedImageUrl(null);
      setRecommendedFurniture([]);
    }
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileDrop(e.dataTransfer.files[0]);
    }
  };
  
  const handleBrowseClick = () => {
    fileInputRef.current.click();
  };
  
  // Функция для определения языка текста
const detectLanguage = (text) => {
  const russianRegex = /[а-яё]/i;
  return russianRegex.test(text) ? 'ru' : 'en';
};

// Функция для перевода текста с русского на английский
const translateText = async (text) => {
  if (!text.trim()) return text;
  
  // Проверяем язык текста
  const language = detectLanguage(text);
  
  // Если текст уже на английском, возвращаем как есть
  if (language === 'en') {
    return text;
  }
  
  try {
    // Используем MyMemory API для перевода
    const response = await fetch(
      `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=ru|en`
    );
    
    if (!response.ok) {
      throw new Error('Translation API error');
    }
    
    const data = await response.json();
    
    // Проверяем успешность перевода
    if (data.responseStatus === 200 && data.responseData.translatedText) {
      return data.responseData.translatedText;
    } else {
      throw new Error('Translation failed');
    }
  } catch (error) {
    console.error('Translation error:', error);
    // В случае ошибки возвращаем оригинальный текст
    return text;
  }
};

// Модифицированная функция handleGenerate
const handleGenerate = async () => {
  if (!previewUrl) {
    alert("Please upload an image first");
    return;
  }
  
  if (isQuickMode && (!roomType || !interiorStyle)) {
    alert("Please select both room type and interior style");
    return;
  }
  
  if (!isQuickMode && !textPrompt.trim()) {
    alert("Please enter a text description");
    return;
  }
  
  // Get user data from localStorage
  const userData = localStorage.getItem("userData");
  if (!userData) {
    alert("User data not found. Please log in again.");
    return;
  }
  
  // Parse user data to get userId
  const userDataObj = JSON.parse(userData);
  const userId = userDataObj.id || userDataObj.userId;
  
  if (!userId) {
    alert("User ID not found. Please log in again.");
    return;
  }
  
  setIsGenerating(true);
  
  try {
    // Create FormData object to send image and other data
    const formData = new FormData();
    formData.append('userId', userId);
    formData.append('image', selectedFile);
    formData.append('is_public', isPublic);
    
    if (isQuickMode) {
      formData.append('mode', 'quick');
      formData.append('roomType', roomType);
      formData.append('interiorStyle', interiorStyle);
    } else {
      formData.append('mode', 'text');
      
      // Переводим текстовый промпт если он на русском
      const translatedPrompt = await translateText(textPrompt);
      console.log('Original prompt:', textPrompt);
      console.log('Translated prompt:', translatedPrompt);
      
      formData.append('textPrompt', translatedPrompt);
    }
    
    // Send request to server
    const response = await fetch('http://127.0.0.1:8000/api/generate', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }
    
    const result = await response.json();
    
    // Process server response
    setGeneratedImageUrl(result.generatedImage);
    setRecommendedFurniture(result.recommendedFurniture || []);
    setIsGenerated(true);
    setSliderPosition(50);
  } catch (error) {
    console.error('Error generating design:', error);
    alert('Failed to generate design. Please try again later.');
  } finally {
    setIsGenerating(false);
  }
};
  
  const handleSliderMouseDown = (e) => {
    e.preventDefault();
    setIsSliderDragging(true);
  };
  
  const handleSliderMouseMove = (e) => {
    if (isSliderDragging && imageContainerRef.current) {
      const containerRect = imageContainerRef.current.getBoundingClientRect();
      const newPosition = ((e.clientX - containerRect.left) / containerRect.width) * 100;
      
      // Constrain between 0 and 100
      setSliderPosition(Math.max(0, Math.min(100, newPosition)));
    }
  };
  
  const handleSliderMouseUp = () => {
    setIsSliderDragging(false);
  };
  
  useEffect(() => {
    document.addEventListener('mousemove', handleSliderMouseMove);
    document.addEventListener('mouseup', handleSliderMouseUp);
    
    return () => {
      document.removeEventListener('mousemove', handleSliderMouseMove);
      document.removeEventListener('mouseup', handleSliderMouseUp);
    };
  }, [isSliderDragging]);

  
  return (
    <div className="design-generator-container">
      <div className="glass-card generator-card">
        <div className="decorative-circle purple"></div>
        <div className="decorative-circle blue"></div>
        
        <h1 className="generator-title">Генератор диазайна интерьера</h1>
        
        <div className="generator-content">
          {/* Image Upload Area */}
          <div 
            className="image-upload-area"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={!previewUrl ? handleBrowseClick : undefined}
            style={{
              height: previewUrl ? 'auto' : '350px',
              cursor: !previewUrl ? 'pointer' : 'default'
            }}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              accept="image/*" 
              style={{ display: 'none' }}
            />
            
            {!previewUrl && (
              <div className="upload-placeholder">
                <div className="upload-icon">
                  <i className="fas fa-cloud-upload-alt"></i>
                </div>
                <p>Перетащите фото сюда</p>
                <p>или</p>
                <button className="browse-button">Выберите Файл</button>
              </div>
            )}
            
            {previewUrl && !isGenerated && (
              <div 
                className="preview-image-container"
                ref={imageContainerRef}
                style={{ 
                  paddingBottom: `${(1 / imageAspectRatio) * 100}%` 
                }}
              >
                <img src={previewUrl} alt="Interior preview" className="preview-image" />
                
                {isGenerating && (
                  <div className="generating-overlay">
                    <div className="loader"></div>
                    <p>Generating your design...</p>
                  </div>
                )}
              </div>
            )}
            
            {isGenerated && generatedImageUrl && (
              <div 
                className="comparison-container"
                ref={imageContainerRef}
                style={{ 
                  paddingBottom: `${(1 / imageAspectRatio) * 100}%` 
                }}
              >
                <div className="comparison-wrapper">
                  {/* Before image - always visible */}
                  <img src={previewUrl} alt="Before" className="comparison-image before-image" />
                  
                  {/* After image - with clip-path to show only the right portion */}
                  <div 
                    className="after-image-container" 
                    style={{ 
                      clipPath: `inset(0 0 0 ${sliderPosition}%)`,
                      width: '100%',
                      height: '100%',
                      position: 'absolute',
                      top: 0,
                      left: 0
                    }}
                  >
                    <img src={generatedImageUrl} alt="After" className="comparison-image after-image" />
                  </div>
                  
                  {/* Slider divider */}
                  <div className="comparison-divider" style={{ left: `${sliderPosition}%` }}>
                    <div 
                      className="slider-handle"
                      onMouseDown={handleSliderMouseDown}
                      onTouchStart={handleSliderMouseDown}
                    >
                      <i className="fas fa-chevron-left"></i>
                      <i className="fas fa-chevron-right"></i>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Mode Selection */}
          <div className="mode-selection">
            <button 
              className={`mode-button ${isQuickMode ? 'active' : ''}`}
              onClick={() => setIsQuickMode(true)}
            >
              Быстрый режим
            </button>
            <button 
              className={`mode-button ${!isQuickMode ? 'active' : ''}`}
              onClick={() => setIsQuickMode(false)}
            >
              Тестовое описание
            </button>
          </div>
          
          {/* Input Form */}
          <div className="input-form">
            {isQuickMode ? (
              <div className="quick-mode-inputs">
                <div className="dropdown-field">
                  <label htmlFor="roomType">Тип Комнаты</label>
                  <select 
                    id="roomType" 
                    value={roomType} 
                    onChange={(e) => setRoomType(e.target.value)}
                    className="select-input"
                  >
                    <option value="">Выберите Тип Комнаты</option>
                    {roomTypes.map((type, index) => (
                      <option key={index} value={type}>{type}</option>
                    ))}
                  </select>
                </div>
                
                <div className="dropdown-field">
                  <label htmlFor="interiorStyle">Стиль интерьера</label>
                  <select 
                    id="interiorStyle" 
                    value={interiorStyle} 
                    onChange={(e) => setInteriorStyle(e.target.value)}
                    className="select-input"
                  >
                    <option value="">Выберите Стиль</option>
                    {interiorStyles.map((style, index) => (
                      <option key={index} value={style}>{style}</option>
                    ))}
                  </select>
                </div>
              </div>
            ) : (
              <div className="text-mode-inputs">
                <div className="text-field">
                  <label htmlFor="textPrompt">Опишите желаемый результат</label>
                  <textarea 
                    id="textPrompt" 
                    value={textPrompt} 
                    onChange={(e) => setTextPrompt(e.target.value)}
                    className="text-input"
                    placeholder="Describe the style, colors, and mood you want for your interior..."
                    rows={4}
                  />
                </div>
              </div>
            )}
                <div>
                <label>
                  <input
                    type="checkbox"
                    checked={isPublic}
                    onChange={handleCheckboxChange}
                    style={{
                      width: '20px',
                      height: '20px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                    }}
                  />
                  Сделать публичным
                </label>
              </div>
            <button 
              className="generate-button"
              onClick={handleGenerate}
              disabled={isGenerating}
            >
              {isGenerating ? 'В процессе...' : 'Начать генерацию'}
            </button>
          </div>
          
          {/* Furniture Recommendations */}
          {isGenerated && recommendedFurniture.length > 0 && (
            <div className="furniture-recommendations">
              <h3>Рекомендуемая мебель</h3>
              <div className="furniture-list">
                {recommendedFurniture.map((item, index) => (
                  <a href={item.url || "#"} className="furniture-item" key={index}>
                    <div className="furniture-image">
                      <img src={item.image} alt={item.name} />
                    </div>
                    <div className="furniture-info">
                      <h4>{item.name}</h4>
                      <p>{item.price}</p>
                    </div>
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}