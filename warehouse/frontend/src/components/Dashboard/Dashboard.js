import React, { useState, useEffect } from 'react';
import SummaryCards from './SummaryCards';
import MetricsChart from './MetricsChart';
import { generateSummaryData, generateMetricsData } from './testingData';
import './Dashboard.css';

const Dashboard = () => {
  const [summaryData, setSummaryData] = useState([]);
  const [metricsData, setMetricsData] = useState([]);
  const [timeRange, setTimeRange] = useState('week'); // Default time range

  useEffect(() => {
    setSummaryData(generateSummaryData());
    setMetricsData(generateMetricsData(timeRange));
  }, [timeRange]);

  return (
    <div className="dashboard">
      <h1>Warehouse Dashboard</h1>
      <div className="time-range-selector">
        <label htmlFor="timeRange">Select Time Range:</label>
        <select
          id="timeRange"
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
        >
          <option value="day">Days</option>
          <option value="week">Weeks</option>
          <option value="month">Months</option>
        </select>
      </div>
      <SummaryCards data={summaryData} />
      <MetricsChart data={metricsData} />
    </div>
  );
};

export default Dashboard;