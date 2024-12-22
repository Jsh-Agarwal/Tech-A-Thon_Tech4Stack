import React from 'react';
import './InventoryGrid.css';

const InventoryGrid = ({ inventory }) => {
  const totalSlots = 50;
  const occupiedSlots = inventory.length;

  return (
    <div className="inventory-grid">
      {Array.from({ length: totalSlots }).map((_, index) => (
        <div
          key={index}
          className={`grid-slot ${index < occupiedSlots ? 'occupied' : ''}`}
        ></div>
      ))}
    </div>
  );
};

export default InventoryGrid;