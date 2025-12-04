# Git Workflow Guide: Committing and Pushing Changes

This guide explains how to commit and push your changes to a git repository, step by step.

## Table of Contents

1. [Checking Current Status](#1-checking-current-status)
2. [Understanding What Needs to Be Committed](#2-understanding-what-needs-to-be-committed)
3. [Handling Ignored Files](#3-handling-ignored-files)
4. [Staging Changes](#4-staging-changes)
5. [Writing a Good Commit Message](#5-writing-a-good-commit-message)
6. [Committing Changes](#6-committing-changes)
7. [Pushing to Remote](#7-pushing-to-remote)
8. [Handling Worktrees](#8-handling-worktrees)
9. [Common Issues and Solutions](#9-common-issues-and-solutions)
10. [Quick Reference Commands](#10-quick-reference-commands)

---

## 1. Checking Current Status

### Check Current Branch

```bash
git branch --show-current
```

**What this does**: Shows the name of the branch you're currently working on.

**Example output**:
```
master
```

### Check Repository Status

```bash
git status
```

**What this does**: Shows:
- Which branch you're on
- Files that have been modified
- Files that are staged for commit
- Files that are untracked (new files not yet in git)
- Whether your branch is ahead/behind the remote

**Example output**:
```
On branch master
Your branch is up to date with 'origin/master'.

Changes not staged for commit:
  modified:   trainer/main.py
  modified:   pong/env/pong_headless.py

Untracked files:
  docs/TRAINING_OPTIMIZATION_JOURNEY.md
  scripts/evaluate_agent.py
```

### Check Remote Repository

```bash
git remote -v
```

**What this does**: Shows where your repository is connected (usually GitHub, GitLab, etc.)

**Example output**:
```
origin	git@github.com:username/repo-name.git (fetch)
origin	git@github.com:username/repo-name.git (push)
```

---

## 2. Understanding What Needs to Be Committed

### Modified Files vs Untracked Files

- **Modified files**: Files that already exist in git but have been changed
- **Untracked files**: New files that git doesn't know about yet

### Check Which Files Will Be Included

```bash
git status --short
```

**Short format codes**:
- `M` = Modified (staged)
- ` M` = Modified (not staged)
- `A` = Added (new file, staged)
- `??` = Untracked (new file, not staged)

---

## 3. Handling Ignored Files

### What Are Ignored Files?

Files listed in `.gitignore` that git will **never** commit (even if they're modified).

**Common examples**:
- Model weights (`.pth`, `.pt`, `.h5`)
- Temporary files
- IDE configuration
- Virtual environments

### Check What's Ignored

```bash
cat .gitignore
```

Or search for specific patterns:
```bash
cat .gitignore | grep "weights"
```

### Files Already Tracked

**Important**: If a file was added to git **before** it was added to `.gitignore`, git will continue tracking it.

**Example problem**:
```bash
git status
# Shows: modified:   final_weights.pth
# But .gitignore has: *.pth
```

**Solution**: Unstage these files:
```bash
git restore --staged final_weights.pth
```

**After this**, git will respect the `.gitignore` rule.

---

## 4. Staging Changes

Staging means telling git "I want to include these changes in my next commit."

### Stage All Changes

```bash
git add -A
```

Or:
```bash
git add .
```

**What this does**: Stages ALL changes (modified and new files) in the current directory and subdirectories.

### Stage Specific Files

```bash
git add path/to/file.py
git add docs/TRAINING_OPTIMIZATION_JOURNEY.md
```

**Stage multiple files**:
```bash
git add file1.py file2.py file3.py
```

**Stage by pattern**:
```bash
git add docs/*.md          # All markdown files in docs/
git add trainer/*.py       # All Python files in trainer/
```

### Stage All Modified Files (Not New)

```bash
git add -u
```

**What this does**: Stages only files that are already tracked by git (not new files).

### Verify What's Staged

```bash
git status
```

**Look for "Changes to be committed"** - these are staged.

### Unstage a File (If You Made a Mistake)

```bash
git restore --staged filename.py
```

Or (older syntax):
```bash
git reset HEAD filename.py
```

---

## 5. Writing a Good Commit Message

A good commit message explains **what** changed and **why**.

### Commit Message Format

```
Short summary (50 chars or less)
<blank line>
Detailed explanation (if needed)
- Bullet point 1
- Bullet point 2
```

### Examples

**Good commit message**:
```
Fix alignment reward scale mismatch

- Corrected alignment reward calculation to match metric scale
- Changed from paddle_half (50px) to screen_height/2 (360px)
- Added gradient rewards for better learning signal
- Expected: Faster Phase 1 convergence
```

**Bad commit message**:
```
fix
```
or
```
changes
```

### Tips

- **Use present tense**: "Fix bug" not "Fixed bug"
- **Be specific**: Mention key files or features changed
- **Explain impact**: What does this change do?
- **Keep first line under 50 characters** (if possible)

---

## 6. Committing Changes

Once files are staged, commit them:

### Simple Commit

```bash
git commit -m "Your commit message here"
```

**Example**:
```bash
git commit -m "Add convergence optimization fixes"
```

### Multi-line Commit Message

```bash
git commit -m "Short summary" -m "Detailed explanation line 1" -m "Line 2"
```

Or use an editor:
```bash
git commit
```
This opens your default editor (vim/nano/vs code) to write a longer message.

### Verify Your Commit

```bash
git log --oneline -1
```

Shows your most recent commit:
```
99dd6ae Implement convergence optimization and comprehensive training improvements
```

### See What Changed

```bash
git show
```

Shows the full diff of your last commit.

---

## 7. Pushing to Remote

After committing, push your changes to the remote repository (GitHub/GitLab).

### Check If You Have Commits to Push

```bash
git status
```

Look for: **"Your branch is ahead of 'origin/master' by X commit(s)"**

### Push to Remote

```bash
git push origin master
```

**Breaking it down**:
- `git push` = send commits to remote
- `origin` = name of your remote (usually "origin")
- `master` = branch name

**Shortcut** (if your branch is already tracking remote):
```bash
git push
```

### Verify Push Success

You should see:
```
To github.com:username/repo-name.git
   982a884..99dd6ae  master -> master
```

This means:
- Pushed from commit `982a884` to `99dd6ae`
- Branch: `master` → `master`

### Check Remote Status

```bash
git status
```

Should show:
```
On branch master
Your branch is up to date with 'origin/master'.
```

---

## 8. Handling Worktrees

### What Are Worktrees?

Git worktrees let you have multiple working directories for the same repository.

### Check Your Worktrees

```bash
git worktree list
```

**Example output**:
```
/Users/user/project          abc1234 [master]
/Users/user/.cursor/worktrees/project/xyz   abc1234 (detached HEAD)
```

### Important Notes

1. **Always commit from the main repository** (not worktrees)
2. **Worktrees share the same git database** - commits in one affect all
3. **Current directory matters** - make sure you're in the main repo

### Check Current Directory

```bash
pwd
```

Make sure you're in your main project directory, not a worktree path.

---

## 9. Common Issues and Solutions

### Issue: "Your branch is behind 'origin/master'"

**Problem**: Remote has changes you don't have locally.

**Solution**:
```bash
git pull origin master
```

This fetches and merges remote changes.

### Issue: "Failed to push - need to pull first"

**Problem**: Remote has commits you don't have.

**Solution**:
```bash
git pull --rebase origin master
git push origin master
```

### Issue: Accidentally Staged Wrong Files

**Problem**: Staged files you don't want to commit.

**Solution**:
```bash
# Unstage everything
git restore --staged .

# Or unstage specific file
git restore --staged filename.py
```

### Issue: Want to Undo Last Commit (Before Push)

**Problem**: Made a commit but want to change it.

**Solution**:
```bash
# Keep changes, uncommit
git reset --soft HEAD~1

# Or discard changes completely
git reset --hard HEAD~1
```

### Issue: Already Pushed, Need to Fix Commit

**Problem**: Pushed a bad commit, need to fix it.

**Solution** (use carefully):
```bash
# Fix and recommit
git commit --amend -m "Corrected message"

# Force push (WARNING: only if you're sure)
git push --force origin master
```

**Warning**: Force push rewrites history. Only do this if you're the only one working on the branch.

---

## 10. Quick Reference Commands

### Status and Information
```bash
git branch --show-current    # Current branch name
git status                    # Full status
git status --short           # Short status
git remote -v                # Remote repository info
git log --oneline -5         # Last 5 commits
```

### Staging
```bash
git add -A                   # Stage everything
git add .                    # Stage everything (current dir)
git add file.py              # Stage specific file
git add -u                   # Stage only modified files
git restore --staged file.py # Unstage file
```

### Committing
```bash
git commit -m "Message"      # Quick commit
git commit                   # Commit with editor
git commit --amend           # Modify last commit
```

### Pushing
```bash
git push origin master       # Push to remote
git push                     # Push (if tracking set)
git pull origin master       # Pull before push
```

### Checking Changes
```bash
git diff                     # Unstaged changes
git diff --staged            # Staged changes
git show                     # Last commit details
```

---

## Complete Workflow Example

Here's a complete example from start to finish:

```bash
# 1. Check where you are
cd /path/to/your/project
git branch --show-current
# Output: master

# 2. See what changed
git status
# Shows: modified files, new files, etc.

# 3. Check ignored files (if needed)
cat .gitignore | grep "weights"
# Output: *.pth

# 4. Stage changes
git add -A

# 5. Check what's staged
git status
# Should show "Changes to be committed"

# 6. Unstage unwanted files (if any)
git restore --staged final_weights.pth

# 7. Commit
git commit -m "Add convergence optimization

- Widen reward window 40% to 60%
- Lower alignment thresholds to 0.35
- Simplify movement rewards
- Expected: Phase 1 in <500 episodes"

# 8. Verify commit
git log --oneline -1
# Output: abc1234 Add convergence optimization

# 9. Push
git push origin master

# 10. Verify
git status
# Should show: "Your branch is up to date with 'origin/master'"
```

---

## Best Practices

1. **Check status frequently**: `git status` before and after each step
2. **Commit often**: Small, focused commits are better than large ones
3. **Write meaningful messages**: Future you will thank you
4. **Test before pushing**: Make sure your code works
5. **Review staged changes**: Use `git diff --staged` before committing
6. **Don't commit sensitive data**: Passwords, API keys, etc.
7. **Don't commit generated files**: Check `.gitignore` is up to date
8. **Pull before push**: Always sync with remote first

---

## Safety Checklist

Before pushing, verify:

- [ ] I'm on the correct branch
- [ ] I'm in the main repository (not a worktree)
- [ ] Only intended files are staged
- [ ] Weight/model files are NOT staged
- [ ] Commit message is clear and descriptive
- [ ] Code has been tested
- [ ] Remote is correct (`git remote -v`)
- [ ] I've pulled latest changes (`git pull` if needed)

---

**Remember**: If you're unsure, `git status` is your friend! It tells you exactly what's happening at each step.

