"use client";

import { useState, useEffect, useRef } from 'react';
import Image from 'next/image';

interface InferenceResponse {
  imageUrl: string;
  bits: string; // "1101", "0011" etc.
}

export default function LiveInferencePage() {
  const [imageUrl, setImageUrl] = useState<string>('');
  const [bits, setBits] = useState<string>('----');
  const [running, setRunning] = useState<boolean>(false);
  const [error, setError] = useState<boolean>(false);
  const intervalRef = useRef<number | null>(null);

  const fetchData = async () => {
    try {
      setError(false);
      // const res = await fetch('/api/inference');
      const res = await fetch('http://localhost:8000/api/camera');
      if (!res.ok) {
        setError(true);
        return;
      }
      const data: InferenceResponse = await res.json();
      setImageUrl(data.imageUrl);
      setBits(data.bits);
    } catch {
      setError(true);
    }
  };

  useEffect(() => {
    if (running) {
      fetchData();
      intervalRef.current = window.setInterval(fetchData, 1000);
    } else if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [running]);

  const handleStart = () => {
    setRunning(true);
  };

  const handleStop = () => {
    setRunning(false);
    setError(false);
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-8 space-y-6">
      <h1 className="text-2xl font-bold">LFBA </h1>
      <h1 className="text-2xl font-bold">リアルタイム推論表示</h1>

      {/* 制御ボタン */}
      <div>
        {running ? (
          <button
            className="px-4 py-2 bg-red-500 text-white rounded"
            onClick={handleStop}
          >
            停止
          </button>
        ) : (
          <button
            className="px-4 py-2 bg-green-500 text-white rounded"
            onClick={handleStart}
          >
            開始
          </button>
        )}
        <button
          className="ml-4 px-4 py-2 border rounded"
          onClick={fetchData}
        >
          手動読み込み
        </button>
      </div>

      {/* エラーメッセージ（表示空間を常に確保） */}
      <div className="h-6 flex items-center">
        {error ? (
          <span className="text-red-500 font-semibold">読み込めません</span>
        ) : (
          <>&nbsp;</>
        )}
      </div>

      {/* カメラ画像表示 */}
      <div className="w-full max-w-md border rounded overflow-hidden">
        {imageUrl ? (
          <Image
            src={imageUrl}
            alt="Inference Image"
            width={640}
            height={480}
            className="object-cover w-full h-auto"
            priority
          />
        ) : (
          <div className="flex items-center justify-center h-48 bg-gray-100">
            <span className="text-gray-500">画像を読み込み中...</span>
          </div>
        )}
      </div>

      {/* ビット表示 */}
      <div className="text-center">
        <span className="text-lg mr-2">推論ビット:</span>
        <span className="inline-block px-4 py-2 bg-gray-200 rounded font-mono text-xl">
          {bits}
        </span>
      </div>
    </main>
  );
}
