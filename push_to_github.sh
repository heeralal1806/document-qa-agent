#!/bin/bash

# Document Q&A AI Agent - GitHub Push Script
# Run this script to push your project to GitHub

echo "=================================================="
echo "  Document Q&A AI Agent - GitHub Push Script"
echo "=================================================="
echo ""

# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    echo "✓ GitHub CLI (gh) found"
    
    # Check if user is logged in
    if gh auth status &> /dev/null; then
        echo "✓ Authenticated with GitHub"
        
        echo ""
        echo "Creating GitHub repository..."
        gh repo create document-qa-agent --public --description "Enterprise-ready AI agent for Document Q&A using LLM APIs"
        
        echo ""
        echo "Setting remote and pushing..."
        git remote add origin https://github.com/$(gh api user --jq '.login')/document-qa-agent.git 2>/dev/null || true
        git branch -M main
        git push -u origin main
        
        echo ""
        echo "✅ Successfully pushed to GitHub!"
        echo "Repository URL: https://github.com/$(gh api user --jq '.login')/document-qa-agent"
    else
        echo "⚠ Not authenticated with GitHub. Please run: gh auth login"
        echo ""
    fi
else
    echo "⚠ GitHub CLI (gh) not installed"
    echo ""
    echo "=================================================="
    echo "  Manual Push Instructions"
    echo "=================================================="
    echo ""
    echo "1. Create a new repository on GitHub:"
    echo "   - Go to https://github.com/new"
    echo "   - Repository name: document-qa-agent"
    echo "   - Description: Enterprise-ready AI agent for Document Q&A"
    echo "   - Public or Private: Your choice"
    echo "   - Do NOT initialize with README, .gitignore, or license"
    echo ""
    echo "2. Run the following commands:"
    echo ""
    echo "   git remote add origin https://github.com/YOUR_USERNAME/document-qa-agent.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "=================================================="
fi

echo ""
echo "Press Enter to exit..."
read

