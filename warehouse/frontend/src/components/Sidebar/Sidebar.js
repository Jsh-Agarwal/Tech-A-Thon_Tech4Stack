import React from 'react';
import { Link } from 'react-router-dom';
import './Sidebar.css';

const Sidebar = () => {
  const navigateToPathPlanning = () => {
    window.location.href = "http://localhost:8502"; 
  };

  const navigateToDetection = () => {
    window.location.href = "http://localhost:8501"; 
  };

  const buttonStyle = {
    width: '100%',
    padding: '12px',
    margin: '5px 0',
    backgroundColor: '#fff', 
    color: '#007bff', 
    border: '1px solid #007bff', 
    textAlign: 'left',
    fontSize: '16px',
    cursor: 'pointer',
    borderRadius: '8px',
    transition: 'background-color 0.3s, transform 0.2s, color 0.2s',
    fontWeight: 'bold',
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
          <button 
            onClick={navigateToPathPlanning} 
            style={buttonStyle}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = '#007bff'; 
              e.target.style.color = '#fff'; 
            }} 
            onMouseOut={(e) => {
              e.target.style.backgroundColor = '#fff'; 
              e.target.style.color = '#007bff'; 
            }}
          >
            Path Planning 
          </button>
        </li>
        <li>
          <button 
            onClick={navigateToDetection} 
            style={buttonStyle}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = '#007bff';
              e.target.style.color = '#fff'; 
            }} 
            onMouseOut={(e) => {
              e.target.style.backgroundColor = '#fff'; 
              e.target.style.color = '#007bff';
            }}
          >
            Box Detection
          </button>
        </li>
      </ul>
    </div>
  );
};

export default Sidebar;