Deployment instructions
=======================

This project is containerized (see `Dockerfile`) and includes a `Procfile` and `render.yaml` for easy deploys.

Option A — Render (recommended)
- Create a free account at https://render.com
- From the Render dashboard choose "New" → "Web Service" → "Connect a repo" (or use the `render.yaml` file to create via the Render dashboard).  
- Pick "Docker" as the environment and point to this repo (or upload zip).  
- Set build/start command (if not auto-filled):  
  - Build: use the default Docker build (no extra command)  
  - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`  
- (Optional) Set environment variables under the service settings: `HF_TOKEN` for faster HF downloads, or any secret keys.  
- Deploy; Render will build the image and expose a public URL you can paste into the challenge endpoint field.

Option B — Fly (container host)
- Install Fly CLI: https://fly.io/docs/getting-started/installing-flyctl/  
- Create an app and deploy:
  ```bash
  fly launch --name relevant-priors --dockerfile Dockerfile
  fly deploy
  ```
- Fly will return a public URL on successful deployment.

Option C — Manual Docker (for testing)  
- Build and run locally (binds to 0.0.0.0:8000):
  ```bash
  docker build -t relevant-priors .
  docker run -p 8000:8000 relevant-priors
  ```

Notes and tips
- Ensure `app/classifier.joblib` and `app/scaler.joblib` are included in the repo (they are saved after running `python train_and_eval.py relevant_priors_public.json`).  
- If your model downloads from Hugging Face on first run, you can set `HF_TOKEN` as an environment variable in your Render/Fly service to avoid rate limits.  
- The evaluator requires a publicly reachable endpoint. After deployment, paste the service URL (e.g., `https://relevant-priors.onrender.com`) into the challenge submission form.
