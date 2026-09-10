# Deploying to Vercel - UPDATED

## What Changed
- Created `api/index.py` for serverless function compatibility
- Added `mangum` adapter for ASGI support
- Embedded HTML interface directly in the API
- Fixed file handling for Vercel's ephemeral storage

## Prerequisites
1. A Vercel account (sign up at https://vercel.com)
2. An OpenAI API key (get one at https://platform.openai.com/api-keys)

## Deployment Steps

### 1. Push Your Code to GitHub
```bash
cd "c:\Users\MOHAN\OneDrive\Desktop\MOHANRAJ R  (RCAS2023BAM051)\rag-resume-bot\rag-resume-bot"
git add .
git commit -m "Fix serverless function configuration for Vercel"
git push origin main
```

### 2. Deploy via Vercel Website

1. Go to https://vercel.com/new
2. Import your GitHub repository (mohanrajramkumar56-cyber/TalentLens)
3. Configure the project:
   - **Framework Preset**: Other
   - **Root Directory**: Leave as default or set to `rag-resume-bot` if needed
   - **Build Command**: (leave empty)
   - **Output Directory**: (leave empty)
4. Add Environment Variable:
   - Click "Environment Variables"
   - Key: `OPENAI_API_KEY`
   - Value: Your OpenAI API key (get from https://platform.openai.com/api-keys)
   - Select all environments (Production, Preview, Development)
5. Click **Deploy**

### 3. After Deployment

Once deployed, you'll get a URL like: `https://your-project.vercel.app`

Visit that URL and you should see the TalentLens interface!

## How to Use

1. **Upload Resumes**: Click "Choose Files" and select PDF resumes
2. **Process**: Click "Upload & Process" button
3. **Ask Questions**: Once status shows "Ready", type questions like:
   - "Which candidate has the most Python experience?"
   - "Who has worked with machine learning?"
   - "Summarize the top 3 candidates"

## Important Notes

- ⚠️ **State is NOT persistent** across serverless invocations. If the function goes idle and restarts, you'll need to re-upload resumes.
- 💰 **OpenAI API costs** apply - monitor usage at https://platform.openai.com/usage
- 📁 **File storage** is temporary - uploaded files are deleted after processing
- 🔄 For production use, consider adding:
  - Redis or database for persistent state
  - S3 or Vercel Blob for file storage
  - Rate limiting and authentication

## Testing Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run locally
python backend.py
```

Visit http://localhost:8000

## Troubleshooting

If you still see errors:
1. Check Vercel logs: Go to your project → Deployments → Click on deployment → View Function Logs
2. Verify `OPENAI_API_KEY` is set correctly in environment variables
3. Make sure all files are pushed to GitHub
4. Try redeploying from Vercel dashboard

## Alternative: Free HuggingFace Models

Want to avoid OpenAI costs? I can help you switch to free HuggingFace models instead!
