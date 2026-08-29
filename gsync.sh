#!/usr/bin/env bash

# gsync.sh - A simple interactive script to sync local changes to a remote Git repository.

# Colors for better visibility
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we are in a Git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo -e "${RED}Error: Not a Git repository.${NC}"
    exit 1
fi

# 1. Show current status
echo -e "${YELLOW}Checking local status...${NC}"
STATUS=$(git status --porcelain)

if [ -z "$STATUS" ]; then
    echo -e "${GREEN}No local changes to commit.${NC}"
else
    echo -e "${YELLOW}Changes detected:${NC}"
    git status -s
    
    # 2. Prompt for commit message
    echo -ne "${YELLOW}Enter commit message (or press Enter for 'Auto-sync'): ${NC}"
    read -r commit_msg
    
    if [ -z "$commit_msg" ]; then
        commit_msg="Auto-sync: $(date +'%Y-%m-%d %H:%M:%S')"
    fi
    
    # 3. Add and Commit
    git add .
    if git commit -m "$commit_msg"; then
        echo -e "${GREEN}\u2713 Changes committed locally.${NC}"
    else
        echo -e "${RED}\u2717 Commit failed.${NC}"
        exit 1
    fi
fi

# 4. Sync with Remote
echo -e "${YELLOW}Syncing with remote repository...${NC}"
BRANCH=$(git branch --show-current)

# Try to push
if git push origin "$BRANCH" 2>/dev/null; then
    echo -e "${GREEN}\u2713 Successfully synced to remote!${NC}"
else
    echo -e "${YELLOW}Push rejected. Attempting to pull and merge remote changes...${NC}"
    
    # Pull changes (merging)
    if git pull origin "$BRANCH" --no-rebase --no-edit; then
        echo -e "${GREEN}\u2713 Remote changes merged.${NC}"
        # Try pushing again after merge
        if git push origin "$BRANCH"; then
            echo -e "${GREEN}\u2713 Successfully synced to remote after merge!${NC}"
        else
            echo -e "${RED}\u2717 Push failed again. Manual intervention may be required.${NC}"
        fi
    else
        echo -e "${RED}\u2717 Merge conflict or pull error. Please resolve manually.${NC}"
    fi
fi#!/usr/bin/env bash

# gsync.sh - A simple interactive script to sync local changes to a remote Git repository.

# Colors for better visibility
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we are in a Git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo -e "${RED}Error: Not a Git repository.${NC}"
    exit 1
fi

# 1. Show current status
echo -e "${YELLOW}Checking local status...${NC}"
STATUS=$(git status --porcelain)

if [ -z "$STATUS" ]; then
    echo -e "${GREEN}No local changes to commit.${NC}"
else
    echo -e "${YELLOW}Changes detected:${NC}"
    git status -s
    
    # 2. Prompt for commit message
    echo -ne "${YELLOW}Enter commit message (or press Enter for 'Auto-sync'): ${NC}"
    read -r commit_msg
    
    if [ -z "$commit_msg" ]; then
        commit_msg="Auto-sync: $(date +'%Y-%m-%d %H:%M:%S')"
    fi
    
    # 3. Add and Commit
    git add .
    if git commit -m "$commit_msg"; then
        echo -e "${GREEN}\u2713 Changes committed locally.${NC}"
    else
        echo -e "${RED}\u2717 Commit failed.${NC}"
        exit 1
    fi
fi

# 4. Sync with Remote
echo -e "${YELLOW}Syncing with remote repository...${NC}"
BRANCH=$(git branch --show-current)

# Try to push
if git push origin "$BRANCH" 2>/dev/null; then
    echo -e "${GREEN}\u2713 Successfully synced to remote!${NC}"
else
    echo -e "${YELLOW}Push rejected. Attempting to pull and merge remote changes...${NC}"
    
    # Pull changes (merging)
    if git pull origin "$BRANCH" --no-rebase --no-edit; then
        echo -e "${GREEN}\u2713 Remote changes merged.${NC}"
        # Try pushing again after merge
        if git push origin "$BRANCH"; then
            echo -e "${GREEN}\u2713 Successfully synced to remote after merge!${NC}"
        else
            echo -e "${RED}\u2717 Push failed again. Manual intervention may be required.${NC}"
        fi
    else
        echo -e "${RED}\u2717 Merge conflict or pull error. Please resolve manually.${NC}"
    fi
fi
