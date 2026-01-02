# Face Blur - Frontend Deployment Guide

## Cloudflare Pages Deployment

This guide covers deploying the Face Blur frontend to Cloudflare Pages for global accessibility.

### Why Cloudflare Pages?

- **Free Tier**: Unlimited bandwidth and requests
- **Global CDN**: Fast loading from anywhere in the world
- **Great Accessibility**: Works well in regions with internet restrictions
- **Easy Deployment**: Git-based automatic deployments
- **Custom Domains**: Free SSL certificates

---

## Prerequisites

- GitHub account
- Cloudflare account (sign up at [cloudflare.com](https://cloudflare.com))
- Backend API URL (from Oracle Cloud deployment)

---

## Step 1: Prepare Frontend Code

### 1.1 Update API URL

Edit `app.js` and update the `API_URL` in the CONFIG object:

```javascript
const CONFIG = {
    // Replace with your Oracle Cloud VM IP or domain
    API_URL: 'http://YOUR_ORACLE_VM_IP:8000',
    // ... rest of config
};
```

Or use a meta tag in `index.html`:

```html
<head>
    <!-- Add before other scripts -->
    <meta name="api-url" content="http://YOUR_ORACLE_VM_IP:8000">
</head>
```

---

## Step 2: Push to GitHub

### 2.1 Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Create new repository: `face-blur-frontend`
3. Set visibility (public for open source)

### 2.2 Push Code

```bash
cd frontend
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/face-blur-frontend.git
git push -u origin main
```

---

## Step 3: Deploy to Cloudflare Pages

### 3.1 Connect Repository

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Go to **Workers & Pages** → **Pages**
3. Click **Create application** → **Pages** → **Connect to Git**
4. Select your GitHub account and repository
5. Click **Begin setup**

### 3.2 Configure Build Settings

- **Project name**: `face-blur` (or your preferred name)
- **Production branch**: `main`
- **Framework preset**: `None`
- **Build command**: (leave empty)
- **Build output directory**: `/`

### 3.3 Environment Variables (Optional)

If you want to set API URL via environment variable:

- Variable name: `API_URL`
- Value: `http://YOUR_ORACLE_VM_IP:8000`

Then update `app.js` to read from it.

### 3.4 Deploy

Click **Save and Deploy**

Wait for the deployment to complete (usually 1-2 minutes).

---

## Step 4: Access Your Application

After deployment, you'll get a URL like:
```
https://face-blur.pages.dev
```

Your app is now live and accessible globally!

---

## Step 5: Custom Domain (Optional)

### 5.1 Add Custom Domain

1. In Cloudflare Pages project settings
2. Go to **Custom domains**
3. Click **Set up a custom domain**
4. Enter your domain: `blur.yourdomain.com`
5. Follow DNS configuration instructions

### 5.2 DNS Configuration

If your domain is on Cloudflare:
- A CNAME record will be added automatically

If your domain is elsewhere:
- Add CNAME record pointing to `face-blur.pages.dev`

---

## Updating the Application

### Automatic Deployments

Every push to the `main` branch triggers an automatic deployment:

```bash
git add .
git commit -m "Update: description of changes"
git push origin main
```

### Manual Deployment

1. Go to Cloudflare Pages dashboard
2. Click on your project
3. Go to **Deployments**
4. Click **Create deployment**

---

## Configuration Options

### API URL Configuration Methods

**Method 1: Hardcoded in app.js**
```javascript
const CONFIG = {
    API_URL: 'https://api.yourdomain.com',
};
```

**Method 2: Meta tag in index.html**
```html
<meta name="api-url" content="https://api.yourdomain.com">
```

**Method 3: Dynamic based on hostname**
```javascript
const CONFIG = {
    API_URL: window.location.hostname === 'localhost' 
        ? 'http://localhost:8000' 
        : 'https://api.yourdomain.com',
};
```

---

## Troubleshooting

### CORS Errors

If you see CORS errors in browser console:

1. Make sure backend CORS is configured correctly in `main.py`:
```python
ALLOWED_ORIGINS = [
    "https://face-blur.pages.dev",  # Your Cloudflare Pages URL
    "https://yourdomain.com",
]
```

2. Restart the backend service:
```bash
sudo systemctl restart faceblur
```

### Mixed Content Errors

If your Cloudflare Pages is HTTPS but API is HTTP:

**Option 1**: Setup HTTPS on your backend (recommended)
- Use Nginx reverse proxy with Let's Encrypt SSL

**Option 2**: Use Cloudflare Tunnel
- Creates secure connection without exposing ports

### Build Failures

Check the build log in Cloudflare dashboard for specific errors.

Common fixes:
- Ensure all files are committed to Git
- Check for syntax errors in JavaScript

---

## Performance Optimization

### Enable Caching

Cloudflare automatically caches static assets. For additional optimization:

1. Go to project **Settings** → **Build & deployments**
2. Add caching headers in `_headers` file:

```
/*
  Cache-Control: public, max-age=31536000
```

### Minification

For production, consider minifying CSS and JavaScript:

```bash
# Install tools
npm install -g terser clean-css-cli

# Minify JS
terser app.js -o app.min.js

# Minify CSS
cleancss style.css -o style.min.css
```

Update `index.html` to use minified files.

---

## Alternative: Using Vite (Optional)

If you prefer a build system:

### Setup Vite Project

```bash
npm create vite@latest frontend -- --template vanilla
cd frontend
npm install
```

### Update wrangler.toml

```toml
name = "face-blur"
compatibility_date = "2024-01-01"

[site]
bucket = "./dist"
```

### Build Settings in Cloudflare

- Build command: `npm run build`
- Output directory: `dist`

---

## Security Considerations

1. **Never commit API keys** to the repository
2. **Use HTTPS** for both frontend and backend
3. **Validate inputs** on both client and server side
4. **Rate limiting** is implemented on backend

---

## Monitoring

### Cloudflare Analytics

1. Go to your project in Cloudflare dashboard
2. Click **Analytics** tab
3. View requests, unique visitors, and more

### Web Vitals

Add performance monitoring:

```html
<script>
  // Basic performance logging
  window.addEventListener('load', () => {
    const timing = performance.timing;
    console.log('Page load time:', timing.loadEventEnd - timing.navigationStart, 'ms');
  });
</script>
```

---

## Support

For issues and questions:
- Create an issue on GitHub
- Check Cloudflare Pages documentation
- Join Cloudflare Discord community
