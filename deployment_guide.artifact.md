# Raka AI Backend Deployment Guide

This guide explains how to deploy your FastAPI backend to a production environment so that your app can work on any user's phone without a local PC connection.

## 1. Prerequisites
- A GitHub account.
- A free account on [Render](https://render.com/) or [Railway](https://railway.app/).
- Your Replicate API Key.

## 2. Prepare the Code
Ensure your `ai_backend` folder contains a `requirements.txt` file. If not, create it:
```bash
pip freeze > requirements.txt
```

## 3. Deploy to Render (Recommended)
1. **Push your code to GitHub**: Create a new repository and upload the contents of the `android` folder (or just `ai_backend`).
2. **Create a New Web Service**:
   - Go to Render Dashboard -> **New** -> **Web Service**.
   - Connect your GitHub repository.
3. **Configure the Service**:
   - **Environment**: `Python`.
   - **Build Command**: `pip install -r requirements.txt`.
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`.
4. **Add Environment Variables**:
   - Go to the **Environment** tab in Render.
   - Add `REPLICATE_API_KEY` with your actual key value.
5. **Deploy**: Render will automatically build and deploy your API.

## 4. Update the Android App
Once deployed, Render will provide a URL like `https://raka-ai-backend.onrender.com/`.

1. Open `E:/AF/app fit/fit/android/app/build.gradle`.
2. Update the `release` block with your new URL:
   ```gradle
   release {
       buildConfigField "String", "AI_VIDEO_BACKEND_URL", "\"https://your-app-name.onrender.com/\""
   }
   ```
3. **Build the APK**: Go to **Build** -> **Build Bundle(s) / APK(s)** -> **Build APK(s)**.
4. **Publish**: This APK is now ready for the Play Store or distribution!

## 5. Production Endpoints
- **Health**: `https://your-app.com/api/health`
- **Text to Video**: `POST https://your-app.com/api/text-to-video`
- **Job Status**: `GET https://your-app.com/api/video-job/{job_id}`
- **Image Gen**: `POST https://your-app.com/api/prompt-to-image`

---
**Note**: Production logs are visible in the Render/Railway dashboard. Use the `RakaProduction` tag to filter them.
