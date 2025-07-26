# Git Commit Command

This command creates git commits following best practices and project guidelines.

## Quick Reference

For comprehensive git commit guidelines, patterns, and examples, see: **[docs/git-best-practices.md](../docs/git-best-practices.md)**

## Essential Commands

### 1. Check Status and Review Changes
```bash
git status
git diff
```

### 2. Stage and Commit
```bash
git add .
git commit -m "<type>(<scope>): <description>"
```

### 3. Common Commit Types
- `feat` - New feature
- `fix` - Bug fix  
- `docs` - Documentation
- `refactor` - Code refactoring
- `test` - Adding tests
- `chore` - Maintenance

## Project-Specific Guidelines

### Automated Footer Policy
**DO NOT** include automated tool footers in commit messages:

```bash
# ❌ AVOID - Don't include these footers:
🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Why**: Keep commit messages clean and focused on the actual changes being made. Tool attribution should not be part of the permanent git history.

**Instead**: Use clean, descriptive commit messages that focus on the changes:

```bash
# ✅ GOOD - Clean, focused commit message:
git commit -m "feat: add user authentication system

Implement JWT-based authentication with:
- Login/logout endpoints
- Token validation middleware
- Password hashing with bcrypt
- Session management"
```

## Examples

```bash
# Stage all changes
git add .

# Commit with conventional format
git commit -m "feat(chains): add RAG chain with memory persistence"

# Push to remote
git push origin feature/rag-implementation
```