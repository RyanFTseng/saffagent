'use client';
import { useState, useEffect } from 'react';

export default function Home() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      console.log('Fetching data from Lambda...');

      const response = await fetch('https://sn3lzs66psvmk3kdixg6djfrne0ghuau.lambda-url.us-west-1.on.aws/', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);

      if (!response.ok) {
        throw new Error(`Lambda responded with status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Lambda response:', result);
      setData(result);
    } catch (error) {
      console.error('Error fetching data:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <h1>Home</h1>
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
        {loading ? (
          <p>Loading data from Lambda...</p>
        ) : error ? (
          <div className="bg-red-100 dark:bg-red-900 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-2 text-red-800 dark:text-red-200">Error:</h3>
            <p className="text-red-700 dark:text-red-300">{error}</p>
            <button
              onClick={fetchData}
              className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Retry
            </button>
          </div>
        ) : (
          <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">Lambda Response:</h3>
            <pre className="text-sm">{JSON.stringify(data, null, 2)}</pre>
          </div>
        )}
      </main>
    </div>
  );
}