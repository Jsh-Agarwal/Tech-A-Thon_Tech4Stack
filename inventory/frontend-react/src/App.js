import React, { useState, useEffect } from 'react';
import UploadForm from './components/UploadForm';
import DetectionResults from './components/DetectionResults';

const App = () => {
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:5000'); // Correct port for Flask backend


    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setDetections(data);
    };

    socket.onopen = () => console.log('WebSocket connected');
    socket.onclose = () => console.log('WebSocket disconnected');
    socket.onerror = (error) => console.error('WebSocket error:', error);

    return () => socket.close();
  }, []);

  return (
    <div>
      <h1>Object Detection App</h1>
      <UploadForm setDetections={setDetections} />
      <DetectionResults detections={detections} />
    </div>
  );
};

export default App;