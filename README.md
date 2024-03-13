# Mitou-Figure2024
未踏の提案システムを構築するためのリポジトリです

![未踏フィギュア_UIイメージ](https://github.com/ryota-takedalab/Mitou-Figure2024/assets/102862947/1cdd2e81-bde6-4534-8542-2c18b22d4b27)



## ローカルのテスト環境
- Webサーバー：Apache2
- フロントエンドサーバー：Next.js, Node.js
- バックエンドサーバー：FastAPI

## 動作環境

### バージョン情報
- python@3.10.10
- Apache/2.4.41 (Ubuntu)
- next@14.1.0
- npm@10.2.4
- node@v20.11.0
- fastapi@0.109.0
- uvicorn@0.27.0.post1
- ffmpeg@1.4
- ffmpeg-python@0.2.0
- aiofiles@23.2.1
- opencv-python@4.9.0.80
- opencv-python-headless@4.9.0.80
- mmdet@3.2.0
- mmpose@1.2.0
- pycalib-simple@2023.12.21.1
- timm@0.9.12
- natsort@-8.4.0
- mmcv@2.1.0
- mmpretrain@1.1.1
- mmengine@0.10.1
- CUDA@12.1
- torch@2.2.0+cu121 　※CUDAとPytorchは自身のGPUのサポートに合わせたバージョンを導入すること

### PC仕様
- OS:Ubuntu 20.04.6 LTS(Windows11 WSL2)
- CPU:12th Gen Intel(R) Core(TM) i5-12400   2.50 GHz
- RAM:16.0 GB
- GPU:NVIDIA GeForce GTX 1660 SUPER

## WEBアプリケーションデモの使い方

### STEP1 Webサーバー(Apache2)の起動
- ターミナル上で`sudo service apache2 start`を実行

### STEP2 バックエンドサーバーの起動
- 実行したいスクリプトのディレクトリ直下で `uvicorn <file_name>:app`を実行
- <file_name>にはファイル名を挿入
- デモ実行の場合は`uvicorn movie:app`

### STEP3 フロントエンドサーバーの起動
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

### STEP4 ローカルホストのページにアクセスし、動画を入力
- ブラウザ上で、`localhost:3000/movie`にアクセス
- コンソール上にエラーが出ている場合はリロード推奨
- `Select Videos`ボタンを押して動画を3つ入力し、処理の終了を待つ(数分かかる)
