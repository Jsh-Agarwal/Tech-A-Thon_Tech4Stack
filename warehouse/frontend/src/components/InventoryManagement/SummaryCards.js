import React from 'react';
import './SummaryCards.css';

const SummaryCards = ({ inventory }) => {
  const totalSlots = 50;
  const occupiedSlots = inventory.length;
  const percentageOccupied = ((occupiedSlots / totalSlots) * 100).toFixed(2);

  return (
    <div className="summary-cards">
      <div className="card occupied">
        <h3>Percentage Occupied</h3>
        <p>{percentageOccupied}%</p>
      </div>
      <div className="card total-packages">
        <h3>Total Packages</h3>
        <p>{inventory.length}</p>
      </div>
      <div className="card total-entries">
        <h3>Total Entries</h3>
        <p>{inventory.length}</p>
      </div>
    </div>
  );
};

export default SummaryCards;