

import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  Legend, CartesianGrid, ResponsiveContainer
} from 'recharts';

const NOAAChart = ({ records }) => {
  if (!records.length) return <p className="text-gray-500 italic">No NOAA data yet</p>;

  // Group records by timestamp
  const dataMap = {};
  records.forEach(r => {
    const ts = new Date(r.timestamp).toLocaleString();
    if (!dataMap[ts]) dataMap[ts] = { timestamp: ts };
    dataMap[ts][r.parameter] = Number(r.value);
  });

  const chartData = Object.values(dataMap).sort(
    (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
  );

  const parameters = [...new Set(records.map(r => r.parameter))];

  // Compute summary stats
  const summary = parameters.map(param => {
    const values = records.filter(r => r.parameter === param).map(r => Number(r.value));
    return {
      parameter: param,
      min: Math.min(...values),
      max: Math.max(...values),
      avg: (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2),
    };
  });

  return (
    <div className="card mb-4">
      <div className="card-body">
        <h3 className="text-lg font-semibold mb-2">NOAA Parameter Trends</h3>

        {/* Summary Panel */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          {summary.map(({ parameter, min, max, avg }) => (
            <div key={parameter} className="bg-blue-50 p-3 rounded shadow-sm border">
              <h4 className="font-medium text-blue-700 capitalize">{parameter.replace("_", " ")}</h4>
              <p className="text-sm text-gray-600">Min: {min}</p>
              <p className="text-sm text-gray-600">Max: {max}</p>
              <p className="text-sm text-gray-600">Avg: {avg}</p>
            </div>
          ))}
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            {parameters.map((param, idx) => (
              <Line
                key={param}
                type="monotone"
                dataKey={param}
                stroke={["#8884d8", "#82ca9d", "#ff7300", "#387908", "#00bcd4", "#e91e63"][idx % 6]}
                strokeWidth={2}
                dot={{ r: 2 }}
                activeDot={{ r: 4 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default NOAAChart;





