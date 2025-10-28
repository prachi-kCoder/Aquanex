// import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer } from "recharts";

// export default function OpenMeteo({ records }) {
//   if (!records?.length) return <p>No Open-Meteo data yet</p>;

//   // Group by parameter
//   const grouped = {};
//   records.forEach(r => {
//     if (!grouped[r.parameter]) grouped[r.parameter] = [];
//     grouped[r.parameter].push({
//       timestamp: new Date(r.timestamp).toLocaleString(),
//       value: r.value,
//       unit: r.unit || ""
//     });
//   });

//   return (
//     <ResponsiveContainer width="100%" height={400}>
//       <LineChart margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
//         <CartesianGrid strokeDasharray="3 3" />
//         <XAxis dataKey="timestamp" />
//         <YAxis />
//         <Tooltip />
//         <Legend />
//         {Object.entries(grouped).map(([param, values], idx) => (
//           <Line
//             key={param}
//             type="monotone"
//             data={values}
//             dataKey="value"
//             name={`${param} (${values[0]?.unit || ""})`}
//             stroke={`hsl(${idx * 60}, 70%, 50%)`}
//             dot={false}
//           />
//         ))}
//       </LineChart>
//     </ResponsiveContainer>
//   );
// }

import React from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  Legend, CartesianGrid, ResponsiveContainer
} from "recharts";

export default function OpenMeteo({ records }) {
  if (!records?.length) return (
    <div className="text-center py-6 text-gray-500 italic">
      No Open-Meteo data available. Try adjusting location or parameters.
    </div>
  );

  // Group by parameter
  const grouped = {};
  records.forEach(r => {
    if (!grouped[r.parameter]) grouped[r.parameter] = [];
    grouped[r.parameter].push({
      timestamp: new Date(r.timestamp).toLocaleString(),
      value: r.value,
      unit: r.unit || "",
    });
  });

  // Summary stats
  const summary = Object.entries(grouped).map(([param, values]) => {
    const nums = values.map(v => Number(v.value));
    return {
      parameter: param,
      unit: values[0]?.unit || "",
      min: Math.min(...nums),
      max: Math.max(...nums),
      avg: (nums.reduce((a, b) => a + b, 0) / nums.length).toFixed(2),
    };
  });

  return (
    <div className="card mb-6">
      <div className="card-body">
        <h2 className="text-xl font-bold text-blue-800 mb-4">🌤️ Open-Meteo Marine Parameters</h2>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 mb-6">
          {summary.map(({ parameter, unit, min, max, avg }) => (
            <div
              key={parameter}
              className="border border-blue-200 rounded-lg p-4 shadow-sm hover:shadow-md transition duration-200 bg-white"
            >
              <h4 className="text-blue-700 font-semibold text-lg capitalize mb-2">
                {parameter.replace(/_/g, " ")}
              </h4>
              <ul className="text-sm text-gray-700 space-y-1">
                <li><strong>Min:</strong> {min} {unit}</li>
                <li><strong>Max:</strong> {max} {unit}</li>
                <li><strong>Avg:</strong> {avg} {unit}</li>
              </ul>
            </div>
          ))}
        </div>

        {/* Chart */}
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-md">
          <ResponsiveContainer width="100%" height={420}>
            <LineChart margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="timestamp" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#f9f9f9", borderRadius: "6px", fontSize: "13px" }}
                labelStyle={{ fontWeight: "bold" }}
              />
              <Legend wrapperStyle={{ fontSize: "13px" }} />
              {Object.entries(grouped).map(([param, values], idx) => (
                <Line
                  key={param}
                  type="monotone"
                  data={values}
                  dataKey="value"
                  name={`${param.replace(/_/g, " ")} (${values[0]?.unit || ""})`}
                  stroke={`hsl(${idx * 60}, 70%, 50%)`}
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

