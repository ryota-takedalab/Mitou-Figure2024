import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid'; // UUIDを生成するためのライブラリ
import Head from 'next/head';

const IndexPage = () => {
  const [videoSrc, setVideoSrc] = useState('');
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [client_id, setClientId] = useState('');

  const fileInputRef = useRef<HTMLInputElement>(null);// ファイル入力要素への参照

  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();// カスタムボタンクリックで実際のファイル入力をトリガー
    }
  };
  
  useEffect(() => {
    // クライアントIDを生成して即座にWebSocket接続を開始する
    const newClientId = uuidv4();
    setClientId(newClientId);
  
    // client_id の状態更新を待つために、WebSocket接続の処理を Promise または useEffect の内部関数で実行する
    const setupWebSocket = (clientId: string) => {
      const websocket = new WebSocket(`${process.env.REACT_APP_WEBSOCKET_URL || 'ws://localhost:8000/ws'}/${clientId}`);
      setWs(websocket);
    
      websocket.onopen = () => console.log('WebSocket Connection opened');
      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received WebSocket message:', data);
        if (data.progress !== undefined) {
          setProgress(data.progress);
        }
      };
      websocket.onerror = (error) => console.error('WebSocket Error:', error);
      websocket.onclose = () => {
        console.log('WebSocket Connection closed');
        setWs(null); // WebSocketの状態を更新
      };
    };    
  
    // clientIdを引数としてsetupWebSocketを呼び出し
    setupWebSocket(newClientId);
  
    // useEffectのクリーンアップ関数でWebSocketを閉じる
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
        const formData = new FormData();
        Array.from(event.target.files).forEach(file => {
            formData.append('files', file);
        });

        setUploading(true);
        setProgress(0); // アップロード開始時に進捗をリセット

        try {
          const response = await fetch(`${process.env.REACT_APP_UPLOAD_URL || 'http://localhost:8000/upload/'}${client_id}`, {
            method: 'POST',
            body: formData,
          });
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const blob = await response.blob();
            const videoUrl = URL.createObjectURL(blob);
            setVideoSrc(videoUrl);
        } catch (error) {
            console.error('Error uploading the files:', error);
        } finally {
            setUploading(false);
        }
    }
};

return (
  <div>
    <Head>
      <title>Mitou 2024 Demo</title>
    </Head>
    <h1 style={{ fontFamily: "'Passion One', cursive" }}>Upload Videos for Pose Estimation</h1>
    <button onClick={handleButtonClick} disabled={uploading} className="customButton">
      {uploading ? 'Uploading...' : 'Select Files'} {/* アップロード中はボタンのテキストを変更 */}
    </button>
    <input
      type="file"
      multiple
      ref={fileInputRef}
      onChange={handleFileChange}
      style={{ display: 'none' }} // 元のファイル入力は非表示
    />
    {uploading && (
      <div>
        <p>Uploading and processing... {progress}%</p>
        <progress value={progress} max="100"></progress>
      </div>
    )}
    {videoSrc && (
      <video controls src={videoSrc} width="720" autoPlay loop>
        Your browser does not support the video tag.
      </video>
    )}
  </div>
 );
};

export default IndexPage;