# Deploying to Vercel

## Prerequisites
1. A Vercel account (sign up at https://vercel.com)
2. An OpenAI API key (get one at https://platform.openai.com/api-keys)

## Deployment Steps

### 1. Push Your Code to GitHub
```bash
git add .
git commit -m "Prepare for Vercel deployment"
git push origin main
```

### 2. Deploy to Vercel

#### Option A: Using Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variable
vercel env add OPENAI_API_KEY
```

#### Option B: Using Vercel Dashboard
1. Go to https://vercel.com/new
2. Import your GitHub repository
3. Configure the project:
   - **Framework Preset**: Other
   - **Build Command**: (leave empty)
   - **Output Directory**: (leave empty)
4. Add Environment Variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your OpenAI API key
5. Click "Deploy"

### 3. Environment Variables Required
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and LLM

## Important Notes

- **Ollama Removed**: The original code used Ollama (local LLM), which doesn't work on serverless platforms like Vercel. This has been replaced with OpenAI's API.
- **Cost**: Using OpenAI API will incur costs based on usage. Monitor your usage at https://platform.openai.com/usage
- **File Storage**: Vercel's serverless functions have ephemeral storage. Uploaded files won't persist between invocations. Consider using external storage (AWS S3, Vercel Blob, etc.) for production.
- **Stateless**: The STATE dictionary in memory won't persist across function invocations. For production, consider using a database or caching service.

## Testing Locally

```bash
# Create .env file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Install dependencies
pip install -r requirements.txt

# Run locally
python backend.py
```

Visit http://localhost:8000 to test.

## Alternative: Use HuggingFace Models (Free)

If you want to avoid OpenAI costs, you can use HuggingFace's free inference API or host your own models. Let me know if you need help with that!
