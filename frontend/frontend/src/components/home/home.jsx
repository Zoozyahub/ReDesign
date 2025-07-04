import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';
import { FaArrowRight, FaRegLightbulb, FaCouch, FaPalette } from 'react-icons/fa';

import intImage from '../../assets/intr.png'; 

const Home = () => {
  const navigate = useNavigate();

  const handleTryNow = () => {
    navigate('/generate');
  };

  return (
    <div className="home-container">
      <div className="home-glass-background">
        <div className="home-decorative-circle circle-1"></div>
        <div className="home-decorative-circle circle-2"></div>
        <div className="home-decorative-circle circle-3"></div>
        
        <div className="home-hero-section">
          <h1 className="home-headline">
            Визуализация интерьера созданная при помощи Искусственного интеллекта, основанная на ваших предпочтениях
          </h1> 
          <div className="home-image-container">
            <div className="home-image-frame">
              <div className="home-image-placeholder">
                {/* Replace with actual image when available */}
                <img 
                  src={intImage} 
                  alt="AI generated interior design visualization" 
                  className="home-showcase-image"
                />
              </div>
            </div>
          </div>
          
          <div className="home-features">
            <div className="home-feature-card">
              <div className="home-feature-icon">
                <FaRegLightbulb />
              </div>
              <h3>Сгенерируем дизайн</h3>
              <p>Искусственный интеллект создаст уникальные концепции интерьера, соответствующие вашим пожеланиям и стилю</p>
            </div>
            
            <div className="home-feature-card">
              <div className="home-feature-icon">
                <FaCouch />
              </div>
              <h3>Подберем варианты мебели</h3>
              <p>Предложим реальные предметы интерьера, доступные для покупки, которые идеально впишутся в созданный дизайн</p>
            </div>
            
            <div className="home-feature-card">
              <div className="home-feature-icon">
                <FaPalette />
              </div>
              <h3>Выберите стиль</h3>
              <p>Минимализм, скандинавский, лофт, классика или модерн - выбирайте любой стиль и получайте интерьер своей мечты</p>
            </div>
          </div>
          
          <div className="home-cta-container">
            <button className="home-cta-button" onClick={handleTryNow}>
              <span>Попробовать сейчас</span>
              <FaArrowRight className="home-cta-icon" />
            </button>
          </div>
        </div>
        
        {/* <div className="home-testimonial-section">
          <div className="home-testimonial-card">
            <div className="home-quote-mark">"</div>
            <p className="home-testimonial-text">
              Я была поражена тем, насколько точно ИИ воплотил мои идеи в визуализации. А самое главное - я смогла легко найти похожую мебель и реализовать проект!
            </p>
            <div className="home-testimonial-author">— Елена Т.</div>
          </div>
        </div> */}
        
        <div className="home-how-it-works">
          <h2 className="home-section-title">Как это работает</h2>
          <div className="home-steps-container">
            <div className="home-step">
              <div className="home-step-number">1</div>
              <h3>Опишите ваши предпочтения</h3>
              <p>Расскажите о стиле, цветах и атмосфере, которую хотите создать</p>
            </div>
            <div className="home-step">
              <div className="home-step-number">2</div>
              <h3>ИИ создаст визуализацию</h3>
              <p>Получите несколько вариантов дизайна интерьера на выбор</p>
            </div>
            <div className="home-step">
              <div className="home-step-number">3</div>
              <h3>Выберите мебель</h3>
              <p>Просмотрите подборку доступных для покупки предметов интерьера</p>
            </div>
          </div>
        </div>
        
        <div className="home-cta-final">
          <h2>Готовы воплотить интерьер своей мечты?</h2>
          <button className="home-cta-button" onClick={handleTryNow}>
            <span>Попробовать сейчас</span>
            <FaArrowRight className="home-cta-icon" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Home;