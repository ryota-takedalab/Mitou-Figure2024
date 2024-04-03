import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid'; // UUIDを生成するためのライブラリ
import Head from 'next/head';
import { urlToUrlWithoutFlightMarker } from 'next/dist/client/components/app-router';

const IndexPage = () => {
  const [videoSrc, setVideoSrc] = useState('');
  const [uploadedVideos, setUploadedVideos] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progressMessage, setProgressMessage] = useState('');
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [client_id, setClientId] = useState('');
  const [dots, setDots] = useState('');
  const [calibrationCompleted, setCalibrationCompleted] = useState(false);



  const fileInputRef = useRef<HTMLInputElement>(null);// ファイル入力要素への参照
  const generateFileInputRef = useRef<HTMLInputElement>(null); // Generate Video用のファイル入力への参照


  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();// ファイル入力をトリガー
    }
  };

  const handleGenerateButtonClick = () => {
    if (generateFileInputRef.current) {
      generateFileInputRef.current.click(); // Generate Videoのファイル入力をトリガー
    }
  };
  
  useEffect(() => {
    const newClientId = uuidv4();
    setClientId(newClientId);
  
    const websocket = new WebSocket(`${process.env.REACT_APP_WEBSOCKET_URL || 'ws://localhost:8000/ws'}/${newClientId}`);
    console.log('WebSocket Connection attempt');
  
    websocket.onopen = () => {
      console.log('WebSocket Connection opened');
      // ハートビートとして定期的にpingを送信
      const heartbeatInterval = setInterval(() => {
        console.log('Sending heartbeat ping');
        websocket.send('ping');
      }, 30000); // 30秒ごと
  
      // Cleanup function for the heartbeat interval
      return () => clearInterval(heartbeatInterval);
    };
  
    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received WebSocket message:', data);
        if (data.status !== undefined) {
          setProgressMessage(data.status);
          console.log(data.status);
        }
      } catch (e) {
        // JSON解析に失敗した場合、テキストメッセージとして処理
        if (event.data === 'pong') {
          console.log('Pong received');
        }
      }
    };
  
    websocket.onerror = (error) => console.error('WebSocket Error:', error);
    websocket.onclose = () => console.log('WebSocket Connection closed');
    setWs(websocket);
  
    const dotsInterval = setInterval(() => {
      setDots(prevDots => (prevDots.length < 3 ? prevDots + '.' : ''));
    }, 500); // 500ミリ秒ごとに更新
  
    // Cleanup function for the WebSocket connection and the dots interval
    return () => {
      if (websocket) {
        websocket.close();
      }
      clearInterval(dotsInterval);
    };
  }, []);

  const handleCalibrateFiles = async (event) => {
    if (event.target.files && event.target.files.length > 0) {
      const formData = new FormData();
      const uploadedUrls = Array.from(event.target.files).map(file => {
        formData.append('files', file);
        return URL.createObjectURL(file); // アップロードされたファイルからURLを生成
      });

      setUploadedVideos(uploadedUrls); // アップロードされた動画のURLを状態に保存
      setUploading(true);
      setCalibrationCompleted(false); 

      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000/calibrate/'}${client_id}`, {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        setCalibrationCompleted(true);
      } catch (error) {
        console.error('Error uploading the files:', error);
      } finally {
        setUploading(false);
      }
    }
  };

  const handleGenerateFiles = async (event) => {
    if (event.target.files && event.target.files.length > 0) {
      const formData = new FormData();
      const uploadedUrls = Array.from(event.target.files).map(file => {
        formData.append('files', file); // 複数のファイルを追加
        return URL.createObjectURL(file); // アップロードされたファイルからURLを生成
      });
  
      setUploading(true);
      setUploadedVideos(uploadedUrls);
      setVideoSrc('');
  
      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000/generate_video/'}${client_id}`, {
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
        console.error('Error during video generation:', error);
      } finally {
        setUploading(false);
      }
    }
  };
  


return (
  <div style={{
      margin: "0",
      padding: "0",
      minHeight: "100vh",
      backgroundColor: '#1C1C1C', 
    }}>
    <Head>
      <title>Mitou 2024 Demo</title>
    </Head>
    <h1 style={{
      fontFamily: "'Passion One', cursive",
      fontSize: "60px",
      textAlign: "center",
      color: 'white',
      backgroundColor: 'black',
      padding: "40px 0"
    }}>
      Upload Videos for Pose Estimation
    </h1>

    <div className="button-container">
      <input
        type="file"
        multiple
        ref={fileInputRef}
        onChange={handleCalibrateFiles}
        style={{ display: 'none' }}
      />
      <button className="customButton" onClick={() => fileInputRef.current.click()} disabled={uploading}>
        Select Videos for Calibration
      </button>

      <input
        type="file"
        multiple
        ref={generateFileInputRef}
        onChange={handleGenerateFiles}
        style={{ display: 'none' }}
        accept="video/*"
      />
      <button className="customButton" onClick={handleGenerateButtonClick} disabled={uploading || uploadedVideos.length === 0}>
        Generate Video
      </button>
    </div>


    {uploading && (
      <div>
        {progressMessage && (
          <div style={{
            fontSize: '35px', // フォントサイズを大きくする
            color: '#333', // フォントカラー（任意で調整可能）
            backgroundColor: '#f0f0f0', // 背景色を設定
            padding: '20px', // 内側の余白を設定
            borderRadius: '10px', // 角を丸くする
            border: '1px solid #ddd', // 境界線を設定
            margin: '10px 0', // 上下の外側の余白を設定
            textAlign: 'center', // テキストを中央揃えにする
            display: 'flex', // 子要素をフレックスボックスとして配置
            justifyContent: 'center', // 子要素を中央に配置
          }}>
            <span>Current Status: {progressMessage}</span>
            <span style={{
              display: 'inline-block', // インラインブロック要素として表示
              minWidth: '30px', // 最小幅を設定してドットの表示部分の幅を固定
              textAlign: 'left', // テキストを左揃えにする
            }}>{dots}</span>
          </div>
        )}
      </div>
    )}
    {calibrationCompleted && (
      <div style={{ marginTop: '50px', textAlign: 'center', color: 'white'}}>
        <p>Calibration completed</p>
      </div>
    )}
    {videoSrc && ( //現在はDetailsもvideoSrcをトリガーにしている
      <div style={{ marginTop: '40px', marginBottom: '50px', display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', gap: '20px', maxWidth: '1200px', width: '100%' }}> {/* コンテナの最大幅を制限 */}
          {/* 動画コンテナ */}
            <div style={{ textAlign: 'center', flex: 2 }}> {/* 動画コンテナにより多くのスペースを割り当て */}
              <p style={{ fontSize: '50px',color: 'white'}}>Result Video</p>
              <video controls src={videoSrc} style={{ maxWidth: "720px", width: '100%', height: "auto"}} autoPlay loop>
                Your browser does not support the video tag.
              </video>
            </div>

          {/* Details見出しをコンテナの外に配置 */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <h2 style={{ fontSize: '50px', textAlign: 'center', marginBottom: '20px',color: 'white'}}>Details</h2>

            {/* Detailsコンテナ */}
            <div style={{ 
              width: '100%', 
              padding: '20px', 
              border: '5px solid grey', 
              borderRadius: '10px', 
              backgroundColor: 'black',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}>
              <div style={{ width: '100%', marginBottom: '20px', borderBottom: '2px solid #ccc'}}>
                <h3 style={{ textAlign: 'center', fontSize: '40px', color: 'white'}}>&lt;Jump&gt;</h3>
                <p style={{ textAlign: 'center', fontSize: '40px', color: 'white'}}>{'1 Lutz' || '-'}</p>
              </div>
              <div style={{ width: '100%', marginBottom: '20px', borderBottom: '2px solid #ccc'}}>
                <h3 style={{ textAlign: 'center', fontSize: '40px', color: 'white'}}>&lt;Rotations&gt;</h3>
                <p style={{ textAlign: 'center', fontSize: '40px', color: 'white'}}>
                {'-31.7° ('} <span style={{ color: '#b71c1c' }}>q</span> {')' || '-'}
                </p>
              </div>
              <div style={{ width: '100%', marginBottom: '20px' }}>
                <h3 style={{ textAlign: 'center', fontSize: '40px', color: 'white'}}>&lt;Edge Angle&gt;</h3>
                <p style={{ textAlign: 'center', fontSize: '40px', color: 'white'}}>
                {'91.2° ('} <span style={{ color: '#b71c1c' }}>!</span> {')' || '-'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    )}




    <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap' }}>
      {uploadedVideos.map((videoSrc, index) => (
        <div key={index} style={{ margin: '10px', textAlign: 'center'}}> {/* 動画の間隔を設定 */}
          <p style={{ margin: "0 auto", fontSize: "30px",color: 'white'}}>{`Uploaded Video ${index + 1}`}</p> {/* 動画の番号を表示 */}
          <video controls src={videoSrc} style={{ width: "500px", display: 'block' }}></video>
        </div>
      ))}
    </div>
  </div>
 );
};

export default IndexPage;