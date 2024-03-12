# Mitou-Figure2024
未踏の提案システムを構築するためのリポジトリです

![未踏フィギュア_UIイメージ](https://github.com/ryota-takedalab/Mitou-Figure2024/assets/102862947/1cdd2e81-bde6-4534-8542-2c18b22d4b27)



## ローカルのテスト環境
- Webサーバー：Apache(XAMPP)
- フロントエンドサーバー：Next.js
- バックエンドサーバー：FastAPI

## 動作環境
- python@3.10.10
- Apache@2.4.58 (Win64)
- next@14cd.1.0
- npm@10.4.0
- node@v20.11.0
- fastapi@0.109.0
- uvicorn@0.27.0.post1
- ffmpeg@1.4
- ffmpeg-python@0.2.0
- aiofiles@23.2.1
- opencv-python@4.9.0.80
- opencv-python-headless@4.9.0.80




## WEBアプリケーションデモの使い方

### STEP1 バックエンドサーバーの起動
- backendフォルダ直下のターミナルで `uvicorn <video_filename>:app --reload`を実行
- '<video_filename>'にはファイル名(movie.pyならmovie)を挿入

### STEP2 フロントエンドサーバーの起動
- `frontend`ディレクトリ直下のターミナルで`npm run dev`を実行 
- デベロッパーツールのコンソール上で`ws://localhost/_next/webpack-hmr' failed: Websocket`のエラーが出たら、`httpd.conf`に以下のコードを追記
```zsh
<Location /_next/webpack-hmr>
    RewriteEngine On
    RewriteCond %{QUERY_STRING} transport=websocket [NC]
    RewriteCond %{HTTP:Upgrade} websocket [NC]
    RewriteCond %{HTTP:Connection} upgrade [NC]
    RewriteRule /(.*) ws://localhost:3000/_next/webpack-hmr/$1 [P,L]
    ProxyPass ws://localhost:3000/_next/webpack-hmr retry=0 timeout=30
    ProxyPassReverse ws://localhost:3000/_next/webpack-hmr
 </Location>
```

### STEP3 動画処理対応
- ffmpegのインストール(本体DL+ PATHを通す+ pip)
- ブラウザ上で動画を読み込むのにH.264であることが必要
