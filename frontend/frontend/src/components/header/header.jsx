import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './header.css';

const Header = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [activeItem, setActiveItem] = useState('Главная');
  const navigate = useNavigate();

  // Эффект для отслеживания прокрутки
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    
    // Очистка слушателя
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  // Имитация проверки авторизации (в реальном проекте здесь должна быть интеграция с вашей системой авторизации)
  useEffect(() => {
    // Пример проверки авторизации
    const checkAuth = () => {
      const userToken = localStorage.getItem('token');
      setIsLoggedIn(!!userToken);
    };
    
    checkAuth();
  }, []);

  // Обработчик клика по пункту меню
  const handleMenuItemClick = (item) => {
  setActiveItem(item);
  // Навигация по пунктам меню
  switch (item) {
    case 'Главная':
      navigate('/');
      break;
    case 'Сгенерировать':
      navigate('/generate');
      break;
    case 'Галерея':
      navigate('/gallery');
      break;
    case 'Партнёры':
      navigate('/partners');
      break;
    default:
      break;
  }
};

  return (
    <header className={`header ${isScrolled ? 'scrolled' : ''}`}>
      <div className="header-container">
        <div className="logo-container">
          <div className="logo">Logo</div>
        </div>
        
        <nav className="main-nav">
          <ul className="menu-items">
            {['Главная', 'Сгенерировать', 'Галерея'].map((item) => (
              <li 
                key={item} 
                className={`menu-item ${activeItem === item ? 'active' : ''}`}
                onClick={() => handleMenuItemClick(item)}
              >
                <span className="menu-text">{item}</span>
                {activeItem === item && <div className="underline"></div>}
              </li>
            ))}
          </ul>
        </nav>
        
        <div className="auth-container">
          {isLoggedIn ? (
            <div className="profile-button"  onClick={() => {navigate('/profile'); setActiveItem('none')}}>
              <span className="profile-text">Профиль</span>
              <div className="profile-icon">
                <div className="avatar"></div>
              </div>
            </div>
          ) : (
            <button className="login-button" onClick={() => {navigate('/login'); setActiveItem('none')}}>
              <span className="login-text">Войти</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;