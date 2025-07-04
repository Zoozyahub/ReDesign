import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./login_form.css";

export default function LoginRegistrationForm() {
  // Инициализируем useNavigate для перенаправления
  const navigate = useNavigate();
  
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);
  const [isConfirmPasswordVisible, setIsConfirmPasswordVisible] = useState(false);
  const [isRegistrationMode, setIsRegistrationMode] = useState(false);
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
    password: "",
    confirmPassword: "",
    country: ""
  });
  const [errors, setErrors] = useState({});
  const [successMessage, setSuccessMessage] = useState(""); // Для отображения сообщения об успехе
  
  const togglePasswordVisibility = () => {
    setIsPasswordVisible(!isPasswordVisible);
  };

  const toggleConfirmPasswordVisibility = () => {
    setIsConfirmPasswordVisible(!isConfirmPasswordVisible);
  };
  
  const toggleMode = () => {
    setIsRegistrationMode(!isRegistrationMode);
    // Reset errors and success message when switching modes
    setErrors({});
    setSuccessMessage("");
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));

    // Clear error for this field when user types
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: null
      }));
    }
  };
  
  const validateForm = () => {
    const newErrors = {};
    
    if (isRegistrationMode) {
      // Registration validation
      if (!formData.firstName.trim()) newErrors.firstName = "First name is required";
      if (!formData.lastName.trim()) newErrors.lastName = "Last name is required";
      if (!formData.email.trim()) {
        newErrors.email = "Email is required";
      } else if (!/^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i.test(formData.email)) {
        newErrors.email = "Invalid email address";
      }
      if (!formData.phone.trim()) newErrors.phone = "Phone number is required";
      if (!formData.password.trim()) {
        newErrors.password = "Password is required";
      } else if (formData.password.length < 8) {
        newErrors.password = "Password must be at least 8 characters";
      }
      if (!formData.confirmPassword.trim()) {
        newErrors.confirmPassword = "Please confirm your password";
      } else if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = "Passwords do not match";
      }
    } else {
      // Login validation
      if (!formData.email.trim()) {
        newErrors.email = "Email or phone number is required";
      }
      if (!formData.password.trim()) {
        newErrors.password = "Password is required";
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  const handleSubmit = async () => {
    if (!validateForm()) return;

    const url = isRegistrationMode
      ? "http://127.0.0.1:8000/api/register"
      : "http://localhost:8000/api/login";

    const payload = isRegistrationMode
      ? {
          name: formData.firstName,
          last_name: formData.lastName,
          email: formData.email,
          phone: formData.phone,
          password: formData.password,
          role: 1
        }
      : {
          email: formData.email,
          password: formData.password,
        };

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        setErrors({ server: errorData.detail || "Server error" });
        return;
      }

      const data = await response.json();
      
      // Проверяем наличие токена в ответе
      if (data.token) {
        // Сохраняем токен в localStorage
        localStorage.setItem("token", data.token);
        console.log('user log in login_form.jsx', data.user)
        localStorage.setItem("userData", JSON.stringify(data.user));
        
        // Показываем сообщение об успехе
        setSuccessMessage(isRegistrationMode ? "Registration successful!" : "Login successful!");
        
        // Перенаправляем на главную страницу после небольшой задержки
        setTimeout(() => {
          navigate("/");
          window.location.reload();
        }, 500);
      } else if (data.access_token) {
        // Проверяем альтернативный ключ для токена (если API возвращает token под другим именем)
        localStorage.setItem("token", data.access_token);
        console.log('user log in login_form.jsx', data.user)
        localStorage.setItem("userData", JSON.stringify(data.user));
        
        // Показываем сообщение об успехе
        setSuccessMessage(isRegistrationMode ? "Registration successful!" : "Login successful!");
        
        // Перенаправляем на главную страницу после небольшой задержки
        setTimeout(() => {
          navigate("/");
          window.location.reload();
        }, 500);
      } else {
        // Если токен отсутствует в ответе, показываем ошибку
        setErrors({ server: "Authentication failed: No token received from server" });
      }
    } catch (error) {
      console.error("Auth error:", error);
      setErrors({ server: "Request failed" });
    }
  };
  
  // posle logina
  useEffect(() => {
    const handleClickOutside = () => {
      // Пусто, но можно добавить нужную логику
    };
    
    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, []);
  
  return (
    <div className="form-container">
      <div className="form-wrapper">
        <div className="glass-card">
          {/* Decorative elements */}
          <div className="decorative-circle purple"></div>
          <div className="decorative-circle blue"></div>
          
          <h1 className="form-title">
            {isRegistrationMode ? "Регистрация" : "Вход"}
          </h1>
          
          <div className="form-content">
            {/* Success Message */}
            {successMessage && (
              <div className="success-message">
                {successMessage}
              </div>
            )}
            
            {/* Registration-only fields */}
            {isRegistrationMode && (
              <>
                {/* First Name Field */}
                <div className="form-field">
                  <label htmlFor="firstName" className="field-label">First Name</label>
                  <div className="input-wrapper">
                    <div className="input-icon">
                      <i className="fas fa-user"></i>
                    </div>
                    <input 
                      type="text" 
                      id="firstName" 
                      className="input-field"
                      placeholder="Enter your first name"
                      value={formData.firstName}
                      onChange={(e) => handleInputChange("firstName", e.target.value)}
                    />
                  </div>
                  {errors.firstName && <p className="error-message">{errors.firstName}</p>}
                </div>
                
                {/* Last Name Field */}
                <div className="form-field">
                  <label htmlFor="lastName" className="field-label">Last Name</label>
                  <div className="input-wrapper">
                    <div className="input-icon">
                      <i className="fas fa-user"></i>
                    </div>
                    <input 
                      type="text" 
                      id="lastName" 
                      className="input-field"
                      placeholder="Enter your last name"
                      value={formData.lastName}
                      onChange={(e) => handleInputChange("lastName", e.target.value)}
                    />
                  </div>
                  {errors.lastName && <p className="error-message">{errors.lastName}</p>}
                </div>
              </>
            )}
            
            {/* Email Field */}
            <div className="form-field">
              <label htmlFor="email" className="field-label">
                {isRegistrationMode ? "Email" : "Email or Phone Number"}
              </label>
              <div className="input-wrapper">
                <div className="input-icon">
                  <i className="fas fa-envelope"></i>
                </div>
                <input 
                  type="text" 
                  id="email" 
                  className="input-field"
                  placeholder={isRegistrationMode ? "Enter your email" : "Enter your email or phone number"}
                  value={formData.email}
                  onChange={(e) => handleInputChange("email", e.target.value)}
                />
              </div>
              {errors.email && <p className="error-message">{errors.email}</p>}
            </div>
            
            {/* Phone Number - Registration only */}
            {isRegistrationMode && (
              <div className="form-field">
                <label htmlFor="phone" className="field-label">Phone Number</label>
                <div className="input-wrapper">
                  <div className="input-icon">
                    <i className="fas fa-phone"></i>
                  </div>
                  <input 
                    type="tel" 
                    id="phone" 
                    className="input-field"
                    placeholder="Enter your phone number"
                    value={formData.phone}
                    onChange={(e) => handleInputChange("phone", e.target.value)}
                  />
                </div>
                {errors.phone && <p className="error-message">{errors.phone}</p>}
              </div>
            )}
            
            {/* Password Field */}
            <div className="form-field">
              <label htmlFor="password" className="field-label">Password</label>
              <div className="input-wrapper">
                <div className="input-icon">
                  <i className="fas fa-lock"></i>
                </div>
                <input 
                  type={isPasswordVisible ? "text" : "password"} 
                  id="password" 
                  className="input-field"
                  placeholder="Enter your password"
                  value={formData.password}
                  onChange={(e) => handleInputChange("password", e.target.value)}
                />
                <button 
                  type="button"
                  className="toggle-password"
                  onClick={togglePasswordVisibility}
                >
                  <i className={`fas ${isPasswordVisible ? "fa-eye-slash" : "fa-eye"}`}></i>
                </button>
              </div>
              {errors.password && <p className="error-message">{errors.password}</p>}
            </div>
            
            {/* Confirm Password Field - Registration only */}
            {isRegistrationMode && (
              <div className="form-field">
                <label htmlFor="confirmPassword" className="field-label">Confirm Password</label>
                <div className="input-wrapper">
                  <div className="input-icon">
                    <i className="fas fa-lock"></i>
                  </div>
                  <input 
                    type={isConfirmPasswordVisible ? "text" : "password"} 
                    id="confirmPassword" 
                    className="input-field"
                    placeholder="Confirm your password"
                    value={formData.confirmPassword}
                    onChange={(e) => handleInputChange("confirmPassword", e.target.value)}
                  />
                  <button 
                    type="button"
                    className="toggle-password"
                    onClick={toggleConfirmPasswordVisibility}
                  >
                    <i className={`fas ${isConfirmPasswordVisible ? "fa-eye-slash" : "fa-eye"}`}></i>
                  </button>
                </div>
                {errors.confirmPassword && <p className="error-message">{errors.confirmPassword}</p>}
              </div>
            )}
 
            {/* Submit Button */}
            <button 
              type="button"
              onClick={handleSubmit}
              className="submit-button"
              disabled={!!successMessage}
            >
              {isRegistrationMode ? "Register" : "Login"}
            </button>
            
            {/* Display server errors */}
            {errors.server && <p className="error-message server-error">{errors.server}</p>}
            
            {/* Footer */}
            <div className="form-footer">
              <p>
                {isRegistrationMode 
                  ? "Already have an account? " 
                  : "Don't have an account? "}
                <a href="#" className="sign-in-link" onClick={(e) => {
                  e.preventDefault(); 
                  toggleMode();
                }}>
                  {isRegistrationMode ? "Sign in" : "Sign up"}
                </a>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}