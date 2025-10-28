// import { useState } from "react";
// import api from "../../services/api";

// const paramsList = [
//   "water_level", "water_temperature", "air_temperature",
//   "wind", "air_pressure", "visibility"
// ];

// export default function NOAAControls({ onFetched }) {
//   const [station, setStation] = useState("");
//   const [selected, setSelected] = useState([]);
//   const [begin, setBegin] = useState("20250101");
//   const [end, setEnd] = useState("20250105");
//   const [loading, setLoading] = useState(false);

//   const toggleParam = (p) =>
//     setSelected((prev) =>
//       prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p]
//     );

//   async function fetchNOAA() {
//     setLoading(true);
//     try {
//       const recs = await api.getNOAARecords(station, selected, begin, end);
//       onFetched(recs);
//     } catch (e) {
//       console.error(e);
//       alert("Failed to fetch NOAA data");
//     } finally {
//       setLoading(false);
//     }
//   }

//   return (
//     <div>
//       <input
//         className="input mb-2"
//         placeholder="Station ID (e.g., 8723214)"
//         value={station}
//         onChange={(e) => setStation(e.target.value)}
//       />
//       <div className="flex gap-2 mb-2">
//         <input
//           type="text"
//           className="input"
//           placeholder="Begin (YYYYMMDD)"
//           value={begin}
//           onChange={(e) => setBegin(e.target.value)}
//         />
//         <input
//           type="text"
//           className="input"
//           placeholder="End (YYYYMMDD)"
//           value={end}
//           onChange={(e) => setEnd(e.target.value)}
//         />
//       </div>
//       <div className="grid grid-cols-2 gap-1 mb-3">
//         {paramsList.map((p) => (
//           <label key={p} className="flex items-center gap-1">
//             <input
//               type="checkbox"
//               checked={selected.includes(p)}
//               onChange={() => toggleParam(p)}
//             />
//             {p}
//           </label>
//         ))}
//       </div>
//       <button className="btn btn-primary w-full" onClick={fetchNOAA} disabled={loading}>
//         {loading ? "Loading..." : "Fetch NOAA Data"}
//       </button>
//     </div>
//   );
// }

import { useState } from "react";
import api from "../../services/api";

const paramsList = [
  "water_level", "water_temperature", "air_temperature",
  "wind", "air_pressure", "visibility", "salinity"
];

export default function NOAAControls({ onFetched }) {
  const [station, setStation] = useState("8723214");
  const [selected, setSelected] = useState(["salinity"]);
  const [begin, setBegin] = useState("20250101");
  const [end, setEnd] = useState("20250105");
  const [loading, setLoading] = useState(false);

  const toggleParam = (p) =>
    setSelected((prev) =>
      prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p]
    );

  async function fetchNOAA() {
    setLoading(true);
    try {
      if (!station || selected.length === 0) {
        alert("Please enter a station ID and select at least one parameter.");
        setLoading(false);
        return;
      }

      const allRecords = [];

      for (const product of selected) {
        const records = await api.ingestNOAA(station, product, begin, end);
        const tagged = records.map(r => ({
          ...r,
          source: "noaa",
          station,
          parameter: product,
          fetchedAt: new Date().toISOString(),
        }));
        allRecords.push(...tagged);
      }

      onFetched(allRecords);
    } catch (e) {
      console.error(e);
      alert("Failed to fetch NOAA data");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <input
        className="input mb-2"
        placeholder="Station ID (e.g., 8723214)"
        value={station}
        onChange={(e) => setStation(e.target.value)}
      />
      <div className="flex gap-2 mb-2">
        <input
          type="text"
          className="input"
          placeholder="Begin (YYYYMMDD)"
          value={begin}
          onChange={(e) => setBegin(e.target.value)}
        />
        <input
          type="text"
          className="input"
          placeholder="End (YYYYMMDD)"
          value={end}
          onChange={(e) => setEnd(e.target.value)}
        />
      </div>
      <div className="grid grid-cols-2 gap-1 mb-3">
        {paramsList.map((p) => (
          <label key={p} className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={selected.includes(p)}
              onChange={() => toggleParam(p)}
            />
            {p}
          </label>
        ))}
      </div>
      <button className="btn btn-primary w-full" onClick={fetchNOAA} disabled={loading}>
        {loading ? "Loading..." : "Fetch NOAA Data"}
      </button>
    </div>
  );
}

