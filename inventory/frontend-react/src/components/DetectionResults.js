import React from 'react';

const DetectionResults = ({ detections }) => {
  return (
    <div>
      <h3>Detection Results:</h3>
      <ul>
        {detections.map((detection, index) => (
          <li key={index}>
            {detection.label} - {Math.round(detection.confidence * 100)}%
          </li>
        ))}
      </ul>
    </div>
  );
};

export default DetectionResults;