import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar/Sidebar';
import Dashboard from './components/Dashboard/Dashboard';
import Tasks from './components/Tasks/Tasks';
import Inventory from './components/Inventory/Inventory';
import PathPlanning from './components/PathPlanning/PathPlanning';
import Detection from './components/Detection/Detection';
import './styles.css';

function App() {
  return (
    <div className="app-container">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/tasks" element={<Tasks />} />
          <Route path="/inventory" element={<Inventory />} />
          <Route path="/path-planning" element={<PathPlanning />} />
          <Route path="/detection" element={<Detection />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;