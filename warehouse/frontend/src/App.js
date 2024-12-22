import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar/Sidebar';
import Dashboard from './components/Dashboard/Dashboard';
import Tasks from './components/Tasks/Tasks';
import PathPlanning from './components/PathPlanning/PathPlanning';
import Detection from './components/Detection/Detection';
import TaskManagement from './components/TaskManagement/TaskManagement';
import InventoryManagement from './components/InventoryManagement/InventoryManagement';

import './styles.css';

function App() {
  return (
    <div className="app-container">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/tasks" element={<Tasks />} /> 
          <Route path="/tasks/manage" element={<TaskManagement />} />  
          <Route path="/inventory" element={<InventoryManagement />} />
          <Route path="/path-planning" element={<PathPlanning />} />
          <Route path="/detection" element={<Detection />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;