# Enable AI - Examples

This folder contains working examples to help you get started with Enable AI.

## 📁 Available Examples

### 1. **simple_usage.py** - Basic Usage
Complete introduction to Enable AI features including:
- Loading API schemas
- Processing natural language queries
- Smart response formatting (auto, table, grouped, chart, concise)
- Working with sample data

**Run:**
```bash
export OPENAI_API_KEY="your-key-here"
python simple_usage.py
```

---

### 2. **streaming_backend.py** + **streaming_frontend.html** - Real-Time Progress
Full-featured FastAPI backend with Server-Sent Events (SSE) for streaming real-time progress updates to frontend.

**Features:**
- Real-time progress updates
- Server-Sent Events (SSE)
- Multiple API endpoints
- Beautiful interactive frontend demo
- Progress stages: parsing, matching, executing, summarizing

**Run:**
```bash
# Start backend
python streaming_backend.py
# Server runs on http://localhost:8000

# Open frontend in browser
open streaming_frontend.html
# Or: python -m http.server 8080
```

**Frontend shows:**
- Progress bar (0-100%)
- Step indicators (✓ completed, ◐ active, ○ pending)
- Real-time messages ("Finding API...", "Calling API...", etc.)
- Final formatted results

---

## 🚀 Quick Start

### Install Dependencies
```bash
pip install enable-ai
# Or with all dependencies:
pip install enable-ai fastapi uvicorn
```

### Set OpenAI Key
```bash
export OPENAI_API_KEY="your-openai-key"
```

### Try Basic Example
```bash
cd examples
python simple_usage.py
```

### Try Streaming Example
```bash
cd examples
python streaming_backend.py
# Then open streaming_frontend.html in browser
```

---

## 📖 What You'll Learn

### From simple_usage.py:
1. How to initialize `APIOrchestrator` and `ResponseFormatter`
2. Process natural language queries
3. Use smart formatting:
   - **Auto** - LLM chooses best format
   - **Table** - Markdown tables for comparisons
   - **Grouped** - Categorized by type/status
   - **Chart** - JSON ready for visualizations
   - **Concise** - Brief summaries
   - **Detailed** - Full breakdowns

### From streaming_backend.py:
1. FastAPI integration with Enable AI
2. Server-Sent Events (SSE) for real-time updates
3. Progress tracking through query lifecycle
4. CORS setup for frontend integration
5. Error handling and graceful degradation

### From streaming_frontend.html:
1. Consuming SSE streams in JavaScript
2. Updating progress bars and UI in real-time
3. Handling different event types (progress, result, error)
4. Modern, responsive UI design

---

## 🔧 Customization

### Use Your Own API
Update `config.json` in the project root:
```json
{
  "api_schemas": [
    {
      "name": "my_api",
      "path": "/path/to/schema.json",
      "type": "openapi"
    }
  ]
}
```

### Modify Streaming Backend
Edit `streaming_backend.py`:
- Change port: `uvicorn.run(app, port=3000)`
- Add authentication
- Customize progress messages
- Add your own endpoints

### Customize Frontend
Edit `streaming_frontend.html`:
- Change colors and styles
- Add your branding
- Modify progress display
- Add more features

---

## 📊 Example Output

### Simple Usage
```
Query: Show items that are low in stock grouped by type

✓ LLM chose format: grouped

Formatted Output:
## Results grouped by Type

### Equipment
- Count: 2
  - STATIONARY MACHINE -8000 AMP (Serial: 98568)
  - STATIONARY MACHINE -3000 AMP (Serial: 98555)

### Consumable
- Count: 2
  - MAGNAFLUX SKL-SP1 (1.0 ml, expires 2028-05-30)
  - MAGNAFLUX MG-2410 (2.0 ml, expires 2028-09-30)
```

### Streaming Progress
```
[0%]   → Starting your request...
[10%]  → Understanding your question...
[20%]  → Intent identified: Get inventory data
[40%]  → Finding the right API...
[50%]  → Found API: Inventory API
[70%]  → Calling GET /inventory/low-stock...
[85%]  → API call completed
[95%]  → Preparing your results...
[100%] → Done! ✓
```

---

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
```bash
export OPENAI_API_KEY="your-key-here"
```

### "No module named 'enable_ai'"
```bash
pip install enable-ai
# Or if running from source:
pip install -e .
```

### Streaming connection fails
- Check backend is running: `curl http://localhost:8000`
- Check CORS settings in `streaming_backend.py`
- Open browser console for errors

### "config.json not found"
Create `config.json` in project root with your API schemas.

---

## 📚 Further Reading

- **Documentation**: See `docs/` folder for detailed guides
- **Smart Formatting**: `docs/smart_formatting.md`
- **Streaming Progress**: `docs/streaming_progress.md`
- **Package README**: `../README.md`

---

## 💡 Tips

1. **Start with simple_usage.py** to understand core concepts
2. **Then try streaming_backend.py** for production-ready patterns
3. **Customize examples** for your specific use case
4. **Check the docs** for advanced features
5. **Read the code** - examples are well-commented!

---

## 🎯 Next Steps

After running these examples:

1. **Configure your API schemas** in `config.json`
2. **Test with your actual APIs**
3. **Integrate into your backend** (FastAPI, Flask, Django, etc.)
4. **Build your frontend** using the streaming example
5. **Deploy to production**

---

## 🤝 Need Help?

- Check the documentation in `docs/`
- Review the inline comments in example code
- Open an issue on GitHub
- Read the main README.md

---

**Happy coding! 🚀**
