import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './profile.css';
import { FaUser, FaPhone, FaEnvelope, FaSignOutAlt } from 'react-icons/fa';

const ProfilePage = () => {
  const [userData, setUserData] = useState({
    firstName: '',
    lastName: '',
    phone: '',
    email: ''
  });
  
  const navigate = useNavigate();

  useEffect(() => {
    // Get user data from localStorage
    const storedUserData = localStorage.getItem("userData");
    console.log('user data v profile', storedUserData.name)
    if (storedUserData) {
      try {
        const parsedData = JSON.parse(storedUserData);
        setUserData(parsedData);
      } catch (error) {
        console.error("Error parsing userData from localStorage:", error);
      }
    } else {
      // Redirect to login if no user data found
      navigate('/login');
    }
  }, [navigate]);

  const handleLogout = () => {
    // Remove token
    localStorage.removeItem("token");
    
    // Remove userData
    localStorage.removeItem("userData");
    
    // Navigate to home route
    navigate('/');
    
    // Reload page to clear header
    window.location.reload();
  };

  return (
    <div className="profile-container">
      <div className="profile-wrapper">
        <div className="profile-card">
          <div className="decorative-circle profile-purple"></div>
          <div className="decorative-circle profile-blue"></div>
          
          <div className="profile-header">
            <div className="profile-avatar">
              {userData.name && userData.last_name 
                ? userData.name.charAt(0) + userData.last_name.charAt(0) 
                : 'UN'}
            </div>
            <h1 className="profile-title">User Profile</h1>
          </div>
          
          <div className="profile-content">
            <div className="profile-field">
              <div className="profile-label">
                <FaUser className="profile-icon" />
                <span>Full Name</span>
              </div>
              <div className="profile-value">
                {userData.name} {userData.last_name}
              </div>
            </div>
            
            <div className="profile-field">
              <div className="profile-label">
                <FaPhone className="profile-icon" />
                <span>Phone Number</span>
              </div>
              <div className="profile-value">
                {userData.phone || 'Not provided'}
              </div>
            </div>
            
            <div className="profile-field">
              <div className="profile-label">
                <FaEnvelope className="profile-icon" />
                <span>Email Address</span>
              </div>
              <div className="profile-value">
                {userData.email || 'Not provided'}
              </div>
            </div>
          </div>
          
          <button className="profile-logout-btn" onClick={handleLogout}>
            <FaSignOutAlt className="logout-icon" />
            <span>Logout</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;