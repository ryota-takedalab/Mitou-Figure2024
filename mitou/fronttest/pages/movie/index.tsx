import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid'; // UUIDを生成するためのライブラリ
import Head from 'next/head';

const IndexPage = () => {
  const [videoSrc, setVideoSrc] = useState('');
  const [uploadedVideos, setUploadedVideos] = useState<string[]>([]);
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
        const uploadedUrls = Array.from(event.target.files).map(file => {
            formData.append('files', file);
            return URL.createObjectURL(file); // アップロードされたファイルからURLを生成
        });

        setUploading(true);
        setProgress(0); // アップロード開始時に進捗をリセット
        setUploadedVideos(uploadedUrls); // アップロードされた動画のURLを状態に保存

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
          setVideoSrc(videoUrl); // 処理後の動画URLをセット
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
      <div style={{ textAlign: 'center'}}>
        <p style={{ fontSize: '50px' }}>Result Video</p>
        <video controls src={videoSrc} style={{width:"720px", margin: "0 auto 80px", display: 'block'}} autoPlay loop>
          Your browser does not support the video tag.
        </video>
      </div>
    )}
    <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap' }}>
      {uploadedVideos.map((videoSrc, index) => (
        <div key={index} style={{ margin: '10px', textAlign: 'center'}}> {/* 動画の間隔を設定 */}
          <p style={{ margin: "0 auto", fontSize: "30px"}}>{`Uploaded Video ${index + 1}`}</p> {/* 動画の番号を表示 */}
          <video controls src={videoSrc} style={{ width: "500px", display: 'block' }}></video>
        </div>
      ))}
    </div>
  </div>
 );
};

export default IndexPage;