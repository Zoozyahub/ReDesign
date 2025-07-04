import React, { useState, useEffect } from 'react';
import './gallery.css';

const Gallery = () => {
  const [isPersonalGallery, setIsPersonalGallery] = useState(false);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState(null);

  useEffect(() => {
    fetchImages();
  }, [isPersonalGallery]);

  const fetchImages = async () => {
    setLoading(true);
    try {
      let url = 'http://localhost:8000/api/images';
      
      if (isPersonalGallery) {
        const userData = localStorage.getItem('userData');
        const userObj = userData ? JSON.parse(userData) : null;
        const userId = userObj ? userObj.id : null;
        
        if (userId) {
          url = `http://localhost:8000/api/images/user/${userId}`;
        } else {
          console.error('User ID not found in localStorage');
          setImages([]);
          setLoading(false);
          return;
        }
      }
      
      // Отправляем реальный запрос к API на localhost:8000
      try {
        const response = await fetch(url);
        
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        setImages(data);
      } catch (apiError) {
        console.error('API fetch error:', apiError);
        
        // Если запрос не удался, показываем моковые данные для тестирования интерфейса
        console.log('Используем тестовые данные, так как API не доступен');
        const mockImages = Array(12).fill().map((_, index) => ({
          id: index + 1,
          author: `Author ${index + 1}`,
          base64_image: 'iVBORw0KGgoAAAANSUhEUgAAASwAAADICAYAAABS39xVAAAFP0lEQVR4Xu3UgQkAMAwCwXb/oV3bFnLAfuBCyTMz5wgQIBCIOMAyzC5AgMCLASwPQYBAJgCwrHUJAgS+AID1EAQIZAIAy1qXIEDgCwCshyBAIBMAWNa6BAECX+ADNC8BZD8tTHEAAAAASUVORK5CYII='
        }));
        
        setImages(mockImages);
      }
      
      setLoading(false);
    } catch (error) {
      console.error('Error fetching images:', error);
      setLoading(false);
    }
  };

//   const handleToggleGallery = () => {
//     setIsPersonalGallery(!isPersonalGallery);
//   };

  const handleImageClick = (image) => {
    setSelectedImage(image);
  };

  const closeImageDetails = () => {
    setSelectedImage(null);
  };

  return (
    <div className="gallery-container">
      <div className="glass-card gallery-header">
        <h1>Фотогалерея</h1>
        <div className="toggle-container">
          <button 
            className={`toggle-btn ${!isPersonalGallery ? 'active' : ''}`}
            onClick={() => setIsPersonalGallery(false)}
          >
            Общая галерея
          </button>
          <button 
            className={`toggle-btn ${isPersonalGallery ? 'active' : ''}`}
            onClick={() => setIsPersonalGallery(true)}
          >
            Моя галерея
          </button>
        </div>
      </div>

      {loading ? (
        <div className="loading-container glass-card">
          <div className="spinner"></div>
          <p>Загрузка изображений...</p>
        </div>
      ) : images.length === 0 ? (
        <div className="no-images glass-card">
          <p>Изображения не найдены</p>
        </div>
      ) : (
        <div className="images-grid">
          {images.map((image) => (
            <div 
              key={image.id} 
              className="image-card glass-card"
              onClick={() => handleImageClick(image)}
            >
              <div className="image-container">
                <img 
                  src={`data:image/jpeg;base64,${image.base64_image}`} 
                  alt={`Изображение ${image.id}`} 
                />
              </div>
              <div className="image-info">
                <h3>Изображение #{image.id}</h3>
                <p className="author">Автор: {image.author}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedImage && (
        <div className="modal-overlay" onClick={closeImageDetails}>
          <div className="modal-content glass-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Изображение #{selectedImage.id}</h2>
              <button className="close-btn" onClick={closeImageDetails}>×</button>
            </div>
            <div className="modal-body">
              <div className="modal-image">
                <img 
                  src={`data:image/jpeg;base64,${selectedImage.base64_image}`} 
                  alt={`Изображение ${selectedImage.id}`} 
                />
              </div>
              <div className="modal-details">
                <p><strong>ID:</strong> {selectedImage.id}</p>
                <p><strong>Автор:</strong> {selectedImage.author}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Gallery;