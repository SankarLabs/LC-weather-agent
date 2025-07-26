# VS Code Local Development Setup

This guide will help you set up and run the Weather Agent locally in VS Code with full debugging capabilities.

## 🚀 Quick Start (Choose One Method)

### Method 1: Native Python Setup (Recommended for Development)
### Method 2: Dev Container (Isolated Environment)
### Method 3: Docker Compose (Full Stack)

---

## 📋 Prerequisites

- **VS Code** with Python extension
- **Python 3.9+** installed locally
- **Git** for version control
- **API Keys**: OpenAI and OpenWeatherMap

## 🔧 Method 1: Native Python Setup (Recommended)

### 1. Clone and Open Project

```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd my-agent-project

# Open in VS Code
code .
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Use VS Code to edit:
code .env
```

**Required in `.env`:**
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Optional but recommended for development
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=weather-agent-dev
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

### 4. Test Installation

```bash
# Test the weather agent
python weather_agent_demo.py

# Test API (in new terminal)
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Test Streamlit (in another terminal)
streamlit run streamlit_app.py
```

### 5. VS Code Python Interpreter

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from your `venv` folder:
   - Windows: `.\venv\Scripts\python.exe`
   - macOS/Linux: `./venv/bin/python`

---

## 🐳 Method 2: Dev Container Setup

### 1. Install Dev Container Extension

Install the "Dev Containers" extension in VS Code.

### 2. Use Dev Container

The project includes a complete dev container setup:

1. Press `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
2. VS Code will build and start the dev container
3. All dependencies will be pre-installed
4. Environment will be automatically configured

---

## 🚢 Method 3: Docker Compose Development

### 1. Install Docker Desktop

Download and install Docker Desktop for your platform.

### 2. Run Development Stack

```bash
# Start development services with hot reload
./scripts/deploy.sh dev

# Or manually:
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f
```

**Access Points:**
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🐛 VS Code Debugging Configuration

The project includes pre-configured debugging setups:

### Debug Configurations Available:

1. **Debug Weather Agent** - Debug the main agent
2. **Debug FastAPI** - Debug the API server
3. **Debug Streamlit** - Debug the UI
4. **Debug Tests** - Debug specific tests

### How to Debug:

1. Set breakpoints in your code
2. Press `F5` or go to Run & Debug panel
3. Select the appropriate debug configuration
4. Start debugging

### Debug Individual Components:

```bash
# Debug weather agent directly
python -m pdb weather_agent_demo.py

# Debug with VS Code integrated terminal
python -c "
from agents.weather_agent import create_weather_agent
from utils.config import get_config
config = get_config()
agent = create_weather_agent(config=config)
response = agent.query('What is the weather in London?')
print(response)
"
```

---

## 🧪 Running Tests in VS Code

### 1. Configure Test Discovery

VS Code should automatically detect pytest tests. If not:

1. Press `Ctrl+Shift+P` → "Python: Configure Tests"
2. Select "pytest"
3. Select root directory

### 2. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_weather_agent.py -v

# Run with coverage
python -m pytest tests/ --cov=agents --cov=tools --cov=utils --cov-report=html

# Run tests in VS Code Test Explorer
# Click the test tube icon in the sidebar
```

### 3. Debug Tests

1. Set breakpoints in test files
2. Right-click on test in Test Explorer
3. Select "Debug Test"

---

## 🔧 Development Workflow

### 1. Daily Development

```bash
# 1. Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Start development servers
# Terminal 1: API server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit UI  
streamlit run streamlit_app.py

# 3. Make changes and test
# Both servers will auto-reload on code changes
```

### 2. Integrated Development

Use VS Code's integrated terminal to run multiple processes:

1. `Ctrl+Shift+`` to open terminal
2. Click "+" to create new terminal tabs
3. Run different services in different tabs

### 3. Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy agents/ tools/ utils/

# Run pre-commit hooks
pre-commit run --all-files
```

---

## 📁 VS Code Workspace Recommendations

### Recommended Extensions:

- **Python** - Essential for Python development
- **Pylance** - Advanced Python language server
- **Black Formatter** - Code formatting
- **autoDocstring** - Generate docstrings
- **GitLens** - Enhanced Git capabilities
- **Thunder Client** - API testing (alternative to Postman)
- **Docker** - Container management
- **YAML** - YAML file support

### Install Extensions:

```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-python.pylance
code --install-extension njpwerner.autodocstring
code --install-extension eamodio.gitlens
code --install-extension rangav.vscode-thunder-client
code --install-extension ms-azuretools.vscode-docker
code --install-extension redhat.vscode-yaml
```

---

## 🔍 Useful VS Code Features

### 1. Integrated Terminal Commands

```bash
# Quick commands in VS Code terminal:

# Test agent functionality
python -c "from agents.weather_agent import create_weather_agent; from utils.config import get_config; agent = create_weather_agent(get_config()); print(agent.query('Weather in Paris?'))"

# Check configuration
python -c "from utils.config import get_config, validate_environment; print(validate_environment())"

# Test individual tools
python -c "from tools.weather_tools import WeatherTool; tool = WeatherTool(); print(tool.run('London'))"
```

### 2. VS Code Tasks

Use `Ctrl+Shift+P` → "Tasks: Run Task" to run:

- **Start Development Stack** - Starts all services
- **Run Tests** - Runs test suite
- **Format Code** - Formats all Python files
- **Build Docker Images** - Builds containers

### 3. Settings Sync

VS Code will automatically configure:
- Python interpreter
- Debugger settings
- Code formatting rules
- Test discovery
- Linting rules

---

## 🚨 Troubleshooting

### Common Issues:

**1. Import Errors**
```bash
# Solution: Ensure VS Code is using the right Python interpreter
# Press Ctrl+Shift+P → "Python: Select Interpreter" → Choose venv interpreter
```

**2. Environment Variables Not Loading**
```bash
# Solution: Restart VS Code after editing .env file
# Or use python-dotenv in your code:
from dotenv import load_dotenv
load_dotenv()
```

**3. Tests Not Discovered**
```bash
# Solution: Configure test framework
# Ctrl+Shift+P → "Python: Configure Tests" → pytest → root directory
```

**4. Debugger Not Working**
```bash
# Solution: Check launch.json configuration
# Ensure correct Python path and working directory
```

**5. Hot Reload Not Working**
```bash
# For FastAPI: Use --reload flag
python -m uvicorn api.main:app --reload

# For Streamlit: Should work automatically
# If not, restart: Ctrl+C and run again
```

### Debug Information:

```bash
# Check Python environment
python --version
pip list | grep langchain
pip list | grep streamlit
pip list | grep fastapi

# Check environment variables
python -c "import os; print('OPENAI_API_KEY:', bool(os.getenv('OPENAI_API_KEY')))"

# Test imports
python -c "
try:
    from agents.weather_agent import create_weather_agent
    print('✅ Weather agent import successful')
except Exception as e:
    print('❌ Import failed:', e)
"
```

---

## 📊 Development Tools Integration

### 1. API Testing with Thunder Client

1. Install Thunder Client extension
2. Create new request collection
3. Test API endpoints:
   - `GET http://localhost:8000/health`
   - `POST http://localhost:8000/query`
   - `GET http://localhost:8000/metrics`

### 2. Git Integration

VS Code has built-in Git support:
- View changes in Source Control panel
- Commit and push directly from VS Code
- Use GitLens for advanced Git features

### 3. Docker Integration

With Docker extension:
- View running containers
- Manage images and volumes
- View container logs
- Execute commands in containers

---

## 🎯 Next Steps

1. **Start with Method 1** (Native Python) for fastest development
2. **Set up debugging** with the provided configurations
3. **Install recommended extensions** for better developer experience
4. **Use integrated terminal** for running multiple services
5. **Test the setup** by running the weather agent demo

## 📞 VS Code Help

- **Command Palette**: `Ctrl+Shift+P` (or `Cmd+Shift+P`)
- **Quick Open**: `Ctrl+P` (or `Cmd+P`)
- **Toggle Terminal**: `Ctrl+`` (or `Cmd+``)
- **Debug Console**: `Ctrl+Shift+Y`
- **Problems Panel**: `Ctrl+Shift+M`

---

**Happy coding! 🚀**