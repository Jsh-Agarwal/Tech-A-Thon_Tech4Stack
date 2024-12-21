import React from 'react';
import './SummaryCards.css';

const SummaryCards = ({ data }) => {
  return (
    <div className="summary-cards">
      {data.map((item, index) => (
        <div className="card" key={index}>
          <span className="icon">{item.icon}</span>
          <h3>{item.value}</h3>
          <p>{item.title}</p>
        </div>
      ))}
    </div>
  );
};

export default SummaryCards;