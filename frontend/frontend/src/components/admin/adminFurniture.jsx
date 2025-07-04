import React, { useState, useEffect } from 'react';
import './AdminFurniture.css';

const AddFurniture = () => {
  const [formData, setFormData] = useState({
    photo: '',
    name: '',
    description: '',
    type: '',
    price: '',
    link: '',
    partner_id: ''
  });
    const [userData, setUserData] = useState({
      firstName: '',
      lastName: '',
      phone: '',
      email: '',
      role: '0'
    });

    useEffect(() => {
    const storedUserData = localStorage.getItem("userData");
    if (storedUserData) {
      try {
        const parsedData = JSON.parse(storedUserData);
        setUserData(parsedData);
      } catch (error) {
        console.error("Error parsing userData from localStorage:", error);
      }
    }
  }, []);

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: files ? files[0] : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = new FormData();
    for (const key in formData) {
      data.append(key, formData[key]);
    }

    try {
      const response = await fetch('http://localhost:8000/api/add-furniture', {
        method: 'POST',
        body: data
      });

      if (response.ok) {
        alert('Мебель успешно добавлена!');
        setFormData({
          photo: '',
          name: '',
          description: '',
          type: '',
          price: '',
          link: '',
          partner_id: ''
        });
      } else {
        alert('Ошибка при добавлении');
      }
    } catch (err) {
      console.error(err);
      alert('Ошибка при отправке запроса');
    }
  };

  return (
  userData?.role == 2 ? (
    <div className="form-container-admin">
      <div className="form-wrapper">
        <div className="glass-card">
          <div className="decorative-circle purple" />
          <div className="decorative-circle blue" />
          <h2 className="form-title">Добавить мебель</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-field">
              <label className="field-label">Фото</label>
              <input
                type="file"
                name="photo"
                className="input-field"
                onChange={handleChange}
              />
            </div>
            <div className="form-field">
              <label className="field-label">Название</label>
              <input
                type="text"
                name="name"
                className="input-field"
                value={formData.name}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-field">
              <label className="field-label">Описание</label>
              <textarea
                name="description"
                className="input-field"
                value={formData.description}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-field">
              <label className="field-label">Тип мебели</label>
              <input
                type="text"
                name="type"
                className="input-field"
                value={formData.type}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-field">
              <label className="field-label">Цена</label>
              <input
                type="number"
                name="price"
                className="input-field"
                value={formData.price}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-field">
              <label className="field-label">Ссылка</label>
              <input
                type="url"
                name="link"
                className="input-field"
                value={formData.link}
                onChange={handleChange}
              />
            </div>
            <div className="form-field">
              <label className="field-label">ID партнёра</label>
              <input
                type="text"
                name="partner_id"
                className="input-field"
                value={formData.partner_id}
                onChange={handleChange}
                required
              />
            </div>
            <button type="submit" className="submit-button">Добавить</button>
          </form>
        </div>
      </div>
    </div>
  ) : (
    <div className="form-container">
      <div className="form-wrapper">
        <div className="glass-card">
          <h2 className="form-title">Страница не найдена</h2>
        </div>
      </div>
    </div>
  )
);
};

export default AddFurniture;
