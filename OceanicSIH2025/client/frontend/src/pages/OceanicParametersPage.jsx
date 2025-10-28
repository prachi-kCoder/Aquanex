// import { useEffect, useMemo, useState } from 'react'
// import api from '../services/api'
// import NOAAChart from '../components/Charts/NOAAChart'
// import NOAAControls from '../components/Controls/NOAAControls'
// import EmptyStateIngest from '../components/common/EmptyStateIngest'

// export default function OceanicParametersPage() {
//   const [records, setRecords] = useState([])
//   const [loading, setLoading] = useState(false)
//   const [error, setError] = useState('')

//   useEffect(() => {
//     let mounted = true
//     async function load() {
//       setLoading(true)
//       setError('')
//       try {
//         const data = await api.getData()
//         if (mounted) setRecords(data)
//       } catch (e) {
//         setError('Failed to load NOAA data')
//       } finally {
//         setLoading(false)
//       }
//     }
//     load()
//     return () => { mounted = false }
//   }, [])

//   const noaaRecords = useMemo(() => records.filter(r => r.source?.toLowerCase().includes('noaa')), [records])

//   return (
//     <div className="container">
//       <div className="card">
//         <div className="card-body">
//           <h2 className="card-title">NOAA Parameters</h2>
//           {error && <div className="text-red-400 mb-2">{error}</div>}
//             <NOAAControls  onFetched={setRecords}/>
//           {noaaRecords.length === 0 ? (
//             <EmptyStateIngest title="No NOAA data found" description="Ingest NOAA data to view parameters and charts." onAfterIngest={async () => setRecords(await api.getData())} />
//           ) : (
//             <NOAAChart records={noaaRecords} />
//           )}
//         </div>
//       </div>
//     </div>
//   )
// }



import { useEffect, useMemo, useState } from 'react';
import api from '../services/api';
import NOAAChart from '../components/Charts/NOAAChart';
import NOAAControls from '../components/Controls/NOAAControls';
import EmptyStateIngest from '../components/common/EmptyStateIngest';

export default function OceanicParametersPage() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      setError('');
      try {
        const data = await api.getData();
        if (mounted) setRecords(data);
      } catch (e) {
        setError('Failed to load NOAA data');
      } finally {
        setLoading(false);
      }
    }
    load();
    return () => { mounted = false };
  }, []);

  const noaaRecords = useMemo(() =>
    records.filter(r => r.source?.toLowerCase() === 'noaa'), [records]);

  return (
    <div className="container">
      <div className="card">
        <div className="card-body">
          <h2 className="card-title">NOAA Parameters</h2>
          {error && <div className="text-red-400 mb-2">{error}</div>}
          <NOAAControls onFetched={setRecords} />
          {noaaRecords.length === 0 ? (
            <EmptyStateIngest
              title="No NOAA data found"
              description="Ingest NOAA data to view parameters and charts."
              onAfterIngest={async () => setRecords(await api.getData())}
            />
          ) : (
            <NOAAChart records={noaaRecords} />
          )}
        </div>
      </div>
    </div>
  );
}
