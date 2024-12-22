import React from 'react';
import { Link } from 'react-router-dom';
import './Sidebar.css';

const Sidebar = () => {
  // Define navigation functions
  const navigateToPathPlanning = () => {
    window.location.href = "http://localhost:8502"; // Replace with your actual URL
  };

  const navigateToDetection = () => {
    window.location.href = "http://localhost:8501"; // Replace with your actual URL
  };

  return (
    <div className="sidebar">
      <h2>Warehouse Management</h2>
      <ul className="sidebar-links">
        <li>
          <Link to="/">Dashboard</Link>
        </li>
        <li>
          <Link to="/tasks/manage">Task Management</Link>
        </li>
        <li>
          <Link to="/inventory">Inventory</Link>
        </li>
        <li>
          <button onClick={navigateToPathPlanning} className="sidebar-button">
            Path Planning (8502)
          </button>
        </li>
        <li>
          <button onClick={navigateToDetection} className="sidebar-button">
            Box Detection (8501)
          </button>
        </li>
      </ul>
    </div>
  );
};

export default Sidebar;
