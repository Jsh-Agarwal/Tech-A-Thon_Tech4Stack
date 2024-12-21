import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './MetricsChart.css';

const MetricsChart = ({ data }) => {
  return (
    <div className="metrics-chart">
      <h2>Warehouse Metrics</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="tasks" stroke="#007bff" name="Tasks" />
          <Line type="monotone" dataKey="boxes" stroke="#ff7300" name="Boxes" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MetricsChart;