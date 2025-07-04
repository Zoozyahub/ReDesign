import './App.css'

import {Routes, Route} from 'react-router-dom'

import LoginRegestrationForm from './components/auth/login_form';
import backgroundImage from './assets/background.jpg'; 
import Header  from './components/header/header';
import InteriorDesignGenerator from './components/generation/generation';
import ProfilePage from './components/profile/profile';
import Home from './components/home/home';
import Gallery from './components/gallery/gallery';
import AdminAddFurniture from './components/admin/adminFurniture';

function App() {
  return (
    <div
      className="App"
      style={{
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: '100% auto', // Ширина фона — 100% контейнера, высота — автоматическая
        backgroundPosition: 'top center', // Фон выравнивается по центру сверху
        backgroundRepeat: 'repeat-y', // Повторение фона только по вертикали (вниз)
      }}>
      <Header/>
      <Routes>
      {/* <Route path={'/'} element={<Home />} /> */}
      <Route path={'/'} element={<div><Home/></div>} />
      <Route path={'/login'} element={<LoginRegestrationForm/>} />
      <Route path={'/generate'} element={<InteriorDesignGenerator/>} />
      <Route path={'/profile'} element={<ProfilePage/>} />
      <Route path={'/gallery'} element={<Gallery/>} />
      <Route path={'/add'} element={<AdminAddFurniture/>} />
      <Route path="*" element={<div>Страница не найдена</div>} />
      </Routes>
    </div>
  )
}

export default App
