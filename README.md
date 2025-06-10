# Vehicle Counting dengan OpenVINO dan MediaMTX

Repositori ini berisi sistem pendeteksi dan penghitung kendaraan real-time berbasis OpenVINO, yang mengalirkan hasilnya melalui RTSP dan HLS menggunakan MediaMTX. Sistem dirancang untuk berjalan optimal di CPU dan disediakan dalam lingkungan Docker.

## Fitur Utama

- Deteksi kendaraan menggunakan model `vehicle-detection-adas-0002` dari OpenVINO.
- Penghitungan jumlah total dan klasifikasi kendaraan secara real-time (mobil kecil, mobil sedang, kendaraan besar).
- Visualisasi langsung di frame video (tanpa penyimpanan metadata terpisah).
- Streaming hasil melalui RTSP (`rtsp://localhost:8554/live`) dan HLS (`http://localhost:8888/live/index.m3u8`).
- Tampilan dashboard berbasis HTML untuk monitoring langsung dari browser.
- Infinite looping video sebagai sumber input.
- Dirancang untuk berjalan sepenuhnya di CPU (tanpa GPU).

## Video Hasil Demo


## Persiapan Awal

### 1. Unduh Model OpenVINO
Unduh model berikut dari Open Model Zoo (FP16) dan simpan di folder `model/`:

- `vehicle-detection-adas-0002.xml`
- `vehicle-detection-adas-0002.bin`

### 2. Siapkan Video Input
Simpan video yang akan digunakan dalam folder `input/` dengan nama `vid.mp4`.

### 3. Build dan Jalankan Container
Jalankan perintah berikut dari root proyek:

```bash
docker-compose up --build

docker-compose down (untuk menghentikan semua service)

| Jenis     | URL                                     |
| --------- | --------------------------------------- |
| RTSP      | `rtsp://localhost:8554/live`            |
| HLS       | `http://localhost:8888/live/index.m3u8` |
| Dashboard | `http://localhost:8080`                 |


