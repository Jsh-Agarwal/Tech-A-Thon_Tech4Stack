import React from 'react';
import { Link } from 'react-router-dom';
import './Sidebar.css';

const Sidebar = () => {
  return (
    <div className="sidebar">
      <h2>Warehouse Management</h2>
      <ul className="sidebar-links">
        <li>
          <Link to="/">Dashboard</Link>
        </li>
        <li>
          <Link to="/tasks">Task Management</Link>
        </li>
        <li>
          <Link to="/inventory">Inventory</Link>
        </li>
        <li>
          <Link to="/path-planning">Path Planning</Link>
        </li>
        <li>
          <Link to="/detection">Box Detection</Link>
        </li>
      </ul>
    </div>
  );
};

export default Sidebar;