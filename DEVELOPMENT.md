# Development Setup Guide

## Quick Development Setup

This guide helps developers get the Stock Ripple Platform running locally for development.

### Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] PostgreSQL 12+ running
- [ ] Neo4j 4.4+ running
- [ ] Git configured
- [ ] IDE/Editor setup (VS Code recommended)

### Step-by-Step Setup

1. **Clone and Setup Environment**
   ```bash
   git clone https://github.com/pnallabe/ripple_data.git
   cd ripple_data
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Database Configuration**
   ```bash
   # PostgreSQL setup
   createdb ripple_db
   psql -d ripple_db -f postgres_schema_ripple.sql
   
   # Neo4j setup - run in Neo4j Browser
   # Copy contents of neo4j_import_ripple.cypher
   ```

3. **Environment Configuration**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your settings
   nano .env
   ```

4. **Verify Installation**
   ```bash
   # Run tests
   python tests/test_platform.py
   
   # Start dashboard
   python main.py --mode dashboard --port 8050
   ```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes and Test**
   ```bash
   # Run tests frequently
   python tests/test_platform.py
   
   # Test specific components
   python -m pytest tests/ -v
   ```

3. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: describe your changes"
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Go to GitHub repository
   - Click "New Pull Request"
   - Select your feature branch
   - Add description and submit

### Troubleshooting

#### Database Connection Issues
```bash
# Check PostgreSQL is running
pg_ctl status

# Check Neo4j is running
neo4j status
```

#### Import Errors
```bash
# Ensure you're in the project directory
pwd

# Check Python path
python -c "import sys; print(sys.path)"
```

#### API Rate Limits
- Free APIs have rate limits
- Consider using caching (Redis) for development
- Use smaller ticker sets for testing

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and testable

### Resources

- [System Design Document](./SDD_Stock_Ripple_Platform.md)
- [Database Schema](./postgres_schema_ripple.sql)
- [Graph Model](./neo4j_import_ripple.cypher)
- [API Documentation](./docs/api.md) (coming soon)