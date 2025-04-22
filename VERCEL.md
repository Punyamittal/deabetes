# Deploying to Vercel

This document explains how to deploy the Diabetes Prediction application to Vercel.

## Prerequisites

1. A Vercel account (sign up at [vercel.com](https://vercel.com))
2. Git installed on your local machine
3. Node.js installed on your local machine

## Deployment Steps

### 1. Push Code to GitHub

Ensure your code is pushed to a GitHub repository:

```bash
git add .
git commit -m "Prepare for Vercel deployment"
git push origin main
```

### 2. Connect to Vercel

1. Login to your Vercel account
2. Click "New Project" button
3. Import your GitHub repository
4. Configure project settings:
   - Framework Preset: `Create React App`
   - Build Command: `npm run build`
   - Output Directory: `build`

### 3. Environment Variables

No environment variables are needed for the basic deployment.

### 4. Deploy

Click "Deploy" button and wait for the deployment to complete.

### 5. Enjoy your deployed application!

After successful deployment, Vercel will provide you with a URL where your application is hosted.

## Project Structure for Vercel

This project is structured to work with Vercel's serverless functions:

- `/api/index.py` - The serverless function that handles backend API requests
- `/vercel.json` - Configuration file for routing and builds
- Frontend React code in the main directory

## Troubleshooting

If you encounter issues with the API:

1. Check the Vercel deployment logs
2. Ensure the model file (`diabetes_model.joblib`) is properly included in your repository
3. Check that your API endpoint in the React code is correctly pointing to `/api/predict`

## Local Development

To test the Vercel setup locally:

1. Install the Vercel CLI: `npm i -g vercel`
2. Run `vercel dev` in the project directory

This will simulate the Vercel environment locally. 