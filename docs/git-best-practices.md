# Git Best Practices

This document outlines git workflow, commit conventions, and collaboration guidelines for the LangChain Context Engineering project.

## 🌊 Git Workflow

### Branch Strategy
We follow a simplified Git Flow with these branch types:

#### Main Branches
- **`main`** - Production-ready code, always deployable
- **`develop`** - Integration branch for features, latest development state

#### Supporting Branches
- **`feature/`** - New features and enhancements
- **`bugfix/`** - Bug fixes for current release
- **`hotfix/`** - Critical fixes for production
- **`docs/`** - Documentation updates
- **`test/`** - Testing improvements

### Branch Naming Convention
```bash
# Features
feature/rag-system-optimization
feature/add-memory-persistence
feature/custom-tool-validation

# Bug fixes
bugfix/agent-timeout-handling
bugfix/memory-leak-in-chains

# Hotfixes
hotfix/security-patch-openai-api
hotfix/critical-rag-performance

# Documentation
docs/update-testing-guide
docs/add-deployment-instructions

# Tests
test/improve-agent-coverage
test/add-performance-benchmarks
```

## 📝 Commit Message Conventions

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- **feat**: New feature or enhancement
- **fix**: Bug fix
- **docs**: Documentation changes
- **test**: Adding or updating tests
- **refactor**: Code refactoring without functionality changes
- **perf**: Performance improvements
- **chore**: Maintenance tasks, dependency updates
- **style**: Code formatting, missing semicolons, etc.
- **ci**: CI/CD pipeline changes

### Scopes (Optional but Recommended)
- **chains**: Chain-related changes
- **agents**: Agent-related changes
- **tools**: Tool implementations
- **memory**: Memory management
- **rag**: RAG system components
- **tests**: Testing infrastructure
- **docs**: Documentation
- **config**: Configuration changes

### Examples

#### Good Commit Messages
```bash
feat(rag): add document preprocessing pipeline

- Implement multi-format document loader (PDF, TXT, HTML)
- Add text chunking with overlap configuration
- Include metadata preservation during processing
- Add comprehensive error handling for unsupported formats

Closes #123

fix(agents): resolve timeout issues in web search tool

- Increase default timeout from 10s to 30s
- Add exponential backoff for failed requests
- Improve error messages for network failures
- Add configuration option for custom timeouts

Fixes #456

test(chains): add performance benchmarks for RAG chains

- Add response time measurements
- Include token usage tracking
- Test with various document sizes
- Set up CI performance regression detection

chore(deps): update langchain to v0.1.5

- Fixes compatibility issues with OpenAI API
- Improves streaming response performance
- Updates breaking changes in memory interfaces

docs(tools): add comprehensive tool development guide

- Include step-by-step tool creation process
- Add best practices for error handling
- Provide real-world examples with code
- Update cross-references in main README
```

#### Poor Commit Messages (Avoid These)
```bash
❌ fix stuff
❌ update code
❌ working on agents
❌ various improvements
❌ WIP
❌ bug fixes
❌ more changes
```

## 🔄 Development Workflow

### 1. Starting New Work
```bash
# Update your local repository
git checkout main
git pull origin main

# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Set up tracking with remote
git push -u origin feature/your-feature-name
```

### 2. During Development
```bash
# Stage specific files (preferred over git add .)
git add path/to/specific/file.py
git add examples/new_feature.py

# Commit with descriptive message
git commit -m "feat(chains): add streaming response support

- Implement streaming callback handler
- Add configuration options for buffer size
- Include error handling for connection issues
- Update chain examples with streaming usage"

# Push regularly to backup your work
git push origin feature/your-feature-name
```

### 3. Before Creating Pull Request
```bash
# Update with latest changes from main
git checkout main
git pull origin main
git checkout feature/your-feature-name
git rebase main

# Run all tests
pytest examples/tests/ -v

# Run linting and formatting
black examples/
flake8 examples/

# Push updated branch
git push origin feature/your-feature-name --force-with-lease
```

### 4. Pull Request Process
1. **Create PR** with descriptive title and detailed description
2. **Link related issues** using "Closes #123" or "Fixes #456"
3. **Request reviews** from relevant team members
4. **Address feedback** and update branch as needed
5. **Merge** once approved (use squash merge for cleaner history)

## 🔍 Code Review Guidelines

### What to Review
- **Functionality**: Does the code work as intended?
- **Testing**: Are there adequate tests for new functionality?
- **Documentation**: Is the code well-documented?
- **Patterns**: Does it follow established project patterns?
- **Performance**: Are there any performance concerns?
- **Security**: Are there any security implications?

### Review Checklist
- [ ] Code follows patterns from `examples/` directory
- [ ] Changes align with `LANGCHAIN_RULES.md` guidelines
- [ ] New functionality includes comprehensive tests
- [ ] Documentation is updated where necessary
- [ ] Error handling is implemented properly
- [ ] Performance impact is acceptable
- [ ] Security best practices are followed

### Providing Feedback
```bash
# Good feedback examples:
✅ "Consider extracting this logic into a separate function for better testability"
✅ "This pattern differs from examples/rag_chain.py - let's maintain consistency"
✅ "Could we add error handling for the case where the API key is invalid?"
✅ "Great implementation! The error handling is very thorough"

# Less helpful feedback:
❌ "This doesn't look right"
❌ "Change this"
❌ "Bad code"
```

## 🚀 Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Workflow
1. **Create release branch** from `develop`
2. **Update version numbers** and changelog
3. **Final testing** and bug fixes
4. **Merge to main** and tag release
5. **Deploy** to production
6. **Merge back** to develop

### Hotfix Process
1. **Create hotfix branch** from `main`
2. **Fix critical issue** with minimal changes
3. **Test thoroughly** in isolation
4. **Merge to main** and tag hotfix release
5. **Cherry-pick** to develop if needed

## 🔧 Git Configuration

### Recommended Git Settings
```bash
# Set up your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Improve merge conflict resolution
git config --global merge.conflictstyle diff3

# Better log formatting
git config --global alias.lg "log --oneline --decorate --graph --all"

# Auto-setup tracking branches
git config --global push.autoSetupRemote true

# Rebase instead of merge for pulls
git config --global pull.rebase true
```

### Useful Git Aliases
```bash
# Add to ~/.gitconfig or use git config --global alias.<name> "<command>"
[alias]
    st = status
    co = checkout
    br = branch
    cm = commit -m
    ps = push
    pl = pull
    lg = log --oneline --decorate --graph --all
    unstage = reset HEAD --
    last = log -1 HEAD
    visual = !gitk
```

## 🔒 Security Considerations

### Sensitive Information
- **Never commit** API keys, passwords, or tokens
- **Use .env files** for local development
- **Use environment variables** in production
- **Review commits carefully** before pushing

### .gitignore Essentials
```gitignore
# Environment variables
.env
.env.local
.env.*.local

# API keys and secrets
**/secrets/
**/*.key
**/*.pem

# Local development
*.log
.vscode/settings.json
.idea/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# Testing
.coverage
htmlcov/
.pytest_cache/

# LangChain specific
chroma_db/
vector_store/
memory_data/
```

## 🐛 Common Issues and Solutions

### Merge Conflicts
```bash
# When conflicts occur during rebase
git status                    # See conflicted files
# Edit files to resolve conflicts
git add <resolved-files>      # Stage resolved files
git rebase --continue         # Continue rebase

# If rebase gets too complex
git rebase --abort           # Abort and try merge instead
git merge main
```

### Accidental Commits
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo specific file from last commit
git reset HEAD~1 <filename>
git commit --amend
```

### Branch Management
```bash
# Delete local branch
git branch -d feature/old-feature

# Delete remote branch
git push origin --delete feature/old-feature

# Rename current branch
git branch -m new-branch-name

# List all branches (including remote)
git branch -a
```

## 📊 Monitoring and Analytics

### Useful Git Commands for Project Health
```bash
# See commit activity
git shortlog -sn --since="1 month ago"

# See file change frequency
git log --pretty=format: --name-only | sort | uniq -c | sort -rg

# See largest files in repository
git ls-tree -r -t -l --full-name HEAD | sort -n -k 4

# See branch divergence
git show-branch --all
```

## 🎯 Integration with Context Engineering

### Git Workflow in Context Engineering
1. **Study examples** in current branch
2. **Create feature branch** for new work
3. **Write INITIAL.md** describing the feature
4. **Use `/generate-lip`** to create implementation plan
5. **Use `/execute-lip`** to implement feature
6. **Commit following conventions** described above
7. **Test thoroughly** using patterns from `examples/tests/`
8. **Create pull request** with detailed description

### Commit Messages for Context Engineering
```bash
# When adding new examples
feat(examples): add custom authentication tool

- Implement OAuth 2.0 authentication pattern
- Include JWT token validation
- Add comprehensive error handling
- Update tool examples README

# When updating LIPs
docs(lips): add deployment considerations to base template

- Include production environment setup
- Add monitoring and logging requirements
- Update validation criteria for production readiness

# When improving context engineering commands
feat(claude): enhance generate-lip with performance analysis

- Add performance benchmarking requirements
- Include resource usage considerations
- Update success criteria with performance metrics
```

Remember: Good git practices support the context engineering approach by maintaining clear history and enabling easy collaboration! 🚀