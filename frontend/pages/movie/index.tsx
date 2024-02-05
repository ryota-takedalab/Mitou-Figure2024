import React, { useState } from 'react';

function MovieUpload() {
  const [videoUrl, setVideoUrl] = useState('');

const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      const formData = new FormData();
      formData.append('file', file);
  
      try {
        const response = await fetch('http://localhost:8000/upload', {
          method: 'POST',
          body: formData,
        });
        if (response.ok) {
          // 変換された動画ファイルをBlobとして受け取り、URLを生成
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          console.log("Video URL:", url);
          setVideoUrl(url);
        } else {
          console.error('Server error', await response.text());
        }
      } catch (error) {
        console.error('Upload failed', error);
      }
    }
  };
  

  return (
    <div>
      <input type="file" onChange={handleFileChange} accept="video/*" />
      <video controls src={videoUrl} style={{ width: '100%', marginTop: '20px' }}
       onError={(e) => console.error("Video loading error:", e)}
       onLoadedMetadata={() => console.log("Video metadata loaded")}
       onLoadedData={() => console.log("Video data loaded")} />
    </div>
  );
}

export default MovieUpload;
